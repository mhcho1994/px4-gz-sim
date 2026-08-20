#!/usr/bin/env python3
r"""Compute axis-aligned projected areas for a Gazebo model.

This command provides two independent estimates of the area exposed to flow
along the model-frame X, Y, and Z axes:

``collision`` mode
    Read box-shaped ``<collision>`` geometries from an SDF file and calculate
    their projected areas analytically.  Link and collision poses are composed,
    so rotated boxes are projected in the containing model frame.  If the model
    contains multiple boxes, their areas are summed without removing overlap.
    The result can consequently overestimate the union of their silhouettes.

``raster`` mode
    Load a triangle mesh, apply a scale and SDF-style pose, project every
    triangle onto the YZ, XZ, and XY planes, and rasterize each silhouette.
    Overlapping triangles occupy the same bitmap pixels and are therefore not
    counted repeatedly.  The result depends slightly on raster resolution and
    should be checked for convergence when high accuracy is required.

The output convention is::

    flow along model X -> projection on YZ
    flow along model Y -> projection on XZ
    flow along model Z -> projection on XY

All geometry dimensions are assumed to be converted to metres before area is
calculated, and all reported areas are in square metres.  The script calculates
geometric projected area only; it does not estimate drag coefficient, porosity,
wake interaction, or an aerodynamic ``CdA``.

Raster defaults reproduce the px4vision ``body.dae`` visual transform used
when this script was created::

    scale       = 0.001, 0.001, 0.001
    translation = 0, 0, 0
    rpy         = 1.5707, 0, 0.01

These defaults are not inferred from the mesh.  Override them when processing
a different visual.  Likewise, the default 10000 pixels/m is a chosen numerical
resolution (0.1 mm/pixel), not information contained in the SDF or mesh.

Examples
--------
Calculate box-collision projections from an SDF::

    python3 tools/auxiliary/compute_projected_area.py \
      --mode collision gz/ardupilot_gazebo/models/px4vision/model.sdf

Select one model when an SDF contains multiple inline models::

    python3 tools/auxiliary/compute_projected_area.py \
      --mode collision world.sdf --model px4vision

Rasterize the px4vision body mesh::

    python3 tools/auxiliary/compute_projected_area.py \
      --mode raster gz/ardupilot_gazebo/models/px4vision/meshes/body.dae \
      --pixels-per-meter 10000 \
      --output-prefix px4vision_projection

Raster mode writes one monochrome PNG for each flow direction.  It requires
``trimesh`` and Pillow in addition to NumPy.  Collision mode only requires
NumPy and the Python standard library.

Limitations
-----------
* Collision mode currently supports only ``<geometry><box>`` elements.
* SDF poses using ``relative_to`` are rejected because resolving the complete
  SDF frame graph is outside this utility's scope.
* Nested models and external ``<include>`` models are not expanded.
* Raster mode takes a mesh directly and does not parse its transform from SDF.
* Neither method accounts for aerodynamic interaction between components.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class Pose:
    """Rigid transform from a local frame into its parent frame.

    ``translation`` is a three-element vector in metres and ``rotation`` is a
    3-by-3 direction cosine matrix.
    """

    translation: np.ndarray
    rotation: np.ndarray


def rotation_matrix_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the SDF fixed-axis rotation Rz(yaw) Ry(pitch) Rx(roll)."""

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=float)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=float)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=float)
    return rz @ ry @ rx


def parse_vector(text: str | None, count: int, description: str) -> np.ndarray:
    """Parse exactly ``count`` whitespace-separated floating-point values."""

    if text is None:
        raise ValueError(f"missing {description}")
    try:
        values = np.asarray([float(value) for value in text.split()], dtype=float)
    except ValueError as exc:
        raise ValueError(f"invalid {description}: {text!r}") from exc
    if len(values) != count:
        raise ValueError(
            f"{description} must contain {count} numbers, got {len(values)}: {text!r}"
        )
    return values


def parse_pose(element: ET.Element) -> Pose:
    """Parse an element's SDF ``<pose>`` or return the identity transform.

    SDF pose values have the order ``x y z roll pitch yaw``.  A ``relative_to``
    attribute is deliberately rejected because this script does not construct
    or resolve a complete SDF semantic pose graph.
    """

    pose_element = element.find("pose")
    if pose_element is None or not pose_element.text or not pose_element.text.strip():
        return Pose(np.zeros(3), np.eye(3))
    if pose_element.get("relative_to"):
        raise ValueError(
            f"pose relative_to={pose_element.get('relative_to')!r} is not supported"
        )
    values = parse_vector(pose_element.text, 6, "pose")
    return Pose(values[:3], rotation_matrix_rpy(*values[3:]))


def compose(parent: Pose, child: Pose) -> Pose:
    """Compose parent-to-model and child-to-parent rigid transforms."""

    return Pose(
        parent.translation + parent.rotation @ child.translation,
        parent.rotation @ child.rotation,
    )


def box_projected_areas(size: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Return a rotated box's YZ, XZ, and XY projection areas.

    For a rectangular box, its orthographic projection along unit vector ``n``
    is the sum of each distinct face area multiplied by the absolute dot
    product between ``n`` and that face's normal.  With ``rotation`` mapping
    local vectors into the model frame, this becomes::

        abs(rotation) @ [sy*sz, sx*sz, sx*sy]

    The returned indices correspond to flow along model X, Y, and Z.
    """

    sx, sy, sz = size
    face_areas = np.array((sy * sz, sx * sz, sx * sy))
    return np.abs(rotation) @ face_areas


def box_collisions(model: ET.Element):
    """Yield box collision geometry and model-frame pose from an SDF model.

    Non-box collision geometries are ignored.  The yielded tuple contains link
    name, collision name, box size, and the composed link/collision pose.
    """

    for link in model.findall("link"):
        link_pose = parse_pose(link)
        for collision in link.findall("collision"):
            box = collision.find("geometry/box")
            if box is None:
                continue
            size_element = box.find("size")
            size = parse_vector(
                size_element.text if size_element is not None else None,
                3,
                "box size",
            )
            if np.any(size <= 0):
                raise ValueError(f"box size must be positive, got {size}")
            yield (
                link.get("name", "<unnamed-link>"),
                collision.get("name", "<unnamed-collision>"),
                size,
                compose(link_pose, parse_pose(collision)),
            )


def select_sdf_model(root: ET.Element, requested_name: str | None) -> ET.Element:
    """Select one inline SDF model, requiring a name if selection is ambiguous."""

    models = [root] if root.tag == "model" else root.findall("model") + root.findall("world/model")
    if requested_name:
        models = [model for model in models if model.get("name") == requested_name]
        if not models:
            raise ValueError(f"model {requested_name!r} was not found")
    elif len(models) > 1:
        names = ", ".join(repr(model.get("name")) for model in models)
        raise ValueError(f"SDF contains multiple models ({names}); select one with --model")
    elif not models:
        raise ValueError("no inline <model> found in the SDF")
    return models[0]


def run_collision(path: Path, model_name: str | None) -> int:
    """Run analytical SDF box projection and print individual and total areas."""

    try:
        root = ET.parse(path).getroot()
        model = select_sdf_model(root, model_name)
        boxes = list(box_collisions(model))
    except (OSError, ET.ParseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not boxes:
        print("No box collision geometries found.")
        return 1

    total = np.zeros(3)
    print(f"SDF:   {path}\nModel: {model.get('name', '<unnamed-model>')}\nFrame: model\n")
    for link_name, collision_name, size, pose in boxes:
        areas = box_projected_areas(size, pose.rotation)
        total += areas
        print(f"{link_name}/{collision_name}")
        print(f"  size: {size[0]:.6g} {size[1]:.6g} {size[2]:.6g} m")
        print(f"  flow X -> YZ: {areas[0]:.6f} m^2")
        print(f"  flow Y -> XZ: {areas[1]:.6f} m^2")
        print(f"  flow Z -> XY: {areas[2]:.6f} m^2")

    if len(boxes) > 1:
        print("\nSum of individual box projections (overlap is not removed)")
        print(f"  flow X -> YZ: {total[0]:.6f} m^2")
        print(f"  flow Y -> XZ: {total[1]:.6f} m^2")
        print(f"  flow Z -> XY: {total[2]:.6f} m^2")
    return 0


def projection_indices(flow_axis: str) -> tuple[int, int]:
    """Return vertex-coordinate indices perpendicular to a flow axis."""

    return {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[flow_axis]


def run_raster(
    path: Path,
    pixels_per_meter: float,
    output_directory: Path,
    output_prefix: str,
    scale: np.ndarray,
    translation: np.ndarray,
    rpy: np.ndarray,
) -> int:
    """Rasterize a transformed mesh silhouette for all three flow directions.

    ``pixels_per_meter`` converts projected metric coordinates to bitmap
    coordinates.  Each occupied pixel represents
    ``1 / pixels_per_meter**2`` square metres.  A separate PNG is written for
    X, Y, and Z flow using ``output_prefix``.
    """

    try:
        import trimesh
        from PIL import Image, ImageDraw
    except ImportError as exc:
        print(f"error: raster mode requires trimesh and Pillow: {exc}", file=sys.stderr)
        return 2

    if pixels_per_meter <= 0:
        print("error: --pixels-per-meter must be positive", file=sys.stderr)
        return 2

    try:
        loaded = trimesh.load(path, force="scene")
        if isinstance(loaded, trimesh.Scene):
            meshes = [g.copy() for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise RuntimeError(f"No triangle mesh found in {path}")
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            raise RuntimeError(f"Unsupported mesh type: {type(loaded)}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: unable to load mesh: {exc}", file=sys.stderr)
        return 2

    vertices = np.asarray(mesh.vertices) * scale
    vertices = (rotation_matrix_rpy(*rpy) @ vertices.T).T + translation
    faces = np.asarray(mesh.faces)
    print(f"Triangles: {len(faces)}")
    print(f"Vertices:  {len(vertices)}")
    print(f"Bounds after SDF transform:\n{np.vstack((vertices.min(0), vertices.max(0)))}")

    for axis, plane in (("x", "YZ"), ("y", "XZ"), ("z", "XY")):
        u, v = projection_indices(axis)
        projected = vertices[:, [u, v]]
        minimum, maximum = projected.min(0), projected.max(0)
        extent = maximum - minimum
        padding = 4
        width = int(math.ceil(extent[0] * pixels_per_meter)) + 2 * padding
        height = int(math.ceil(extent[1] * pixels_per_meter)) + 2 * padding
        image = Image.new("1", (width, height), 0)
        draw = ImageDraw.Draw(image)

        for face in faces:
            triangle = projected[face]
            if abs(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) < 1e-14:
                continue
            pixel = (triangle - minimum) * pixels_per_meter
            pixel[:, 0] += padding
            pixel[:, 1] = height - padding - pixel[:, 1]
            draw.polygon([tuple(point) for point in pixel], fill=1)

        area = np.count_nonzero(np.asarray(image)) / pixels_per_meter**2
        output_directory = output_directory.resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        output = output_directory / f"{output_prefix}_flow_{axis}.png"
        image.save(output)
        print(f"Flow along {axis.upper()}: {plane} area = {area:.6f} m^2 ({output})")
    return 0


def vector_argument(value: str) -> np.ndarray:
    """Parse a CLI vector written as three comma-separated numbers."""

    try:
        result = np.asarray([float(item) for item in value.split(",")], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers") from exc
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return result


def parse_args() -> argparse.Namespace:
    """Define and parse the common collision/raster command-line interface."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("collision", "raster"),
        required=True,
        help="analytical SDF boxes or rasterized mesh silhouette",
    )
    parser.add_argument(
        "--input", 
        type=Path, 
        help="SDF file (collision) or DAE mesh file (raster)")

    parser.add_argument("--model", help="model name for an SDF containing multiple models")
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=10000.0,
        help="raster resolution (default: 10000, or 0.1 mm/pixel)",
    )
    parser.add_argument(
        "--output-prefix",
        default="px4vision_projection",
        help="prefix for raster output PNGs (default: px4vision_projection)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("."),
        help="directory for raster output PNGs (default: current directory)",
    )
    parser.add_argument(
        "--scale",
        type=vector_argument,
        default=np.array((0.001, 0.001, 0.001)),
        help="raster mesh XYZ scale as x,y,z (default: 0.001,0.001,0.001)",
    )
    parser.add_argument(
        "--translation",
        type=vector_argument,
        default=np.zeros(3),
        help="raster mesh XYZ translation in metres (default: 0,0,0)",
    )
    parser.add_argument(
        "--rpy",
        type=vector_argument,
        default=np.array((1.5707, 0.0, 0.01)),
        help="raster mesh SDF roll,pitch,yaw in radians (default: 1.5707,0,0.01)",
    )

    args = parser.parse_args()

    if args.mode == "collision" and args.input.suffix.lower() != ".sdf":
        parser.error("--mode collision requires an .sdf input file")

    if args.mode == "raster" and args.input.suffix.lower() != ".dae":
        parser.error("--mode raster requires a .dae input file")

    return args


def main() -> int:
    """Dispatch to the selected projection method and return a process status."""

    args = parse_args()
    if args.mode == "collision":
        return run_collision(args.input, args.model)
    return run_raster(
        args.input,
        args.pixels_per_meter,
        args.output_directory,
        args.output_prefix,
        args.scale,
        args.translation,
        args.rpy,
    )


if __name__ == "__main__":
    raise SystemExit(main())
