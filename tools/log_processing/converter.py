#!/usr/bin/env python3
"""
Convert PX4 ULog and ArduPilot BIN logs into CSV files for one run directory
or for all batch runs under a data root.

Expected directory layout
-------------------------
<data_root>/
  run_000/
    px4_logs/
      *.ulg
    ardu_logs/
      logs/
        000001.BIN
  run_001/
    ...

Outputs
-------
PX4:
  /data/run_000/px4_logs/processed/*.csv

ArduPilot:
  /data/run_000/ardu_logs/processed/*.csv

Usage examples
--------------
Process one run:
    python convert.py --run-dir /data/run_000

Process all runs:
    python convert.py --data-root /data

PX4 only:
    python convert.py --data-root /data --px4-only

ArduPilot only:
    python convert.py --run-dir /data/run_000 --ardu-only

Skip already processed directories:
    python convert.py --data-root /data --skip-existing
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from pyulog.ulog2csv import convert_ulog2csv


def convert_px4_ulog_to_csv(px4_logs_dir: Path, skip_existing: bool = False) -> list[Path]:
    """
    Convert all PX4 ULog files in a run directory into CSV files.

    This function searches for ``*.ulg`` files under:
        <run_dir>/px4_logs/

    and writes the converted CSV files under:
        <run_dir>/px4_logs/processed/

    Parameters
    ----------
    px4_logs_dir : Path
        Path to the PX4 log directory, e.g. ``/data/run_000/px4_logs``.
    skip_existing : bool, optional
        If True and the processed directory already contains CSV files,
        conversion is skipped.

    Returns
    -------
    list[Path]
        A sorted list of generated CSV files in the processed directory.

    Notes
    -----
    - Uses ``pyulog.ulog2csv.convert_ulog2csv(...)``.
    - All ULog files found in the directory are converted.
    - Each ULog typically expands into multiple CSV files, one per topic.
    """
    px4_logs_dir = Path(px4_logs_dir)
    logs_dir = px4_logs_dir / "raw"
    processed_dir = px4_logs_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    existing_csvs = sorted(processed_dir.glob("*.csv"))
    if skip_existing and existing_csvs:
        print(f"[SKIP] PX4 conversion already exists: {processed_dir}")
        return existing_csvs

    ulg_files = sorted(logs_dir.glob("*.ulg"))
    if not ulg_files:
        print(f"[WARN] No PX4 ULog files found in: {logs_dir}")
        return []

    for ulg_file in ulg_files:
        print(f"[INFO] Converting PX4 ULog: {ulg_file}")
        # Convert the ULog into one or more CSV files in processed_dir.
        convert_ulog2csv(
            str(ulg_file),   # ulog_file_name
            None,            # messages: convert all messages
            str(processed_dir),  # output directory
            ",",             # delimiter
            None,            # time_s
            None,            # time_e
            False,           # disable_str_exceptions
        )

    return sorted(processed_dir.glob("*.csv"))


def convert_ardupilot_bin_to_csv(ardu_logs_dir: Path, skip_existing: bool = False) -> list[Path]:
    """
    Convert all ArduPilot BIN log files in a run directory into CSV files.

    This function searches for ``*.BIN`` files under:
        <run_dir>/ardu_logs/logs/

    and writes the converted CSV files under:
        <run_dir>/ardu_logs/processed/

    Parameters
    ----------
    ardu_logs_dir : Path
        Path to the ArduPilot log directory, e.g. ``/data/run_000/ardu_logs``.
    skip_existing : bool, optional
        If True and the processed directory already contains CSV files,
        conversion is skipped.

    Returns
    -------
    list[Path]
        A sorted list of generated CSV files in the processed directory.

    Raises
    ------
    RuntimeError
        If the ``bin2csv`` executable is not available or conversion fails.

    Notes
    -----
    - Public pybinlog documentation exposes ``bin2csv`` as a command-line tool,
      so this function calls that installed entry point directly.
    - All ``*.BIN`` files in ``ardu_logs/logs`` are converted.
    """
    ardu_logs_dir = Path(ardu_logs_dir)
    logs_dir = ardu_logs_dir / "raw" / "logs"
    processed_dir = ardu_logs_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    existing_csvs = sorted(processed_dir.glob("*.csv"))
    if skip_existing and existing_csvs:
        print(f"[SKIP] ArduPilot conversion already exists: {processed_dir}")
        return existing_csvs

    if shutil.which("bin2csv") is None:
        raise RuntimeError(
            "Could not find 'bin2csv' in PATH. "
            "Please install pybinlog so that the CLI entry point is available."
        )

    bin_files = sorted(logs_dir.glob("*.BIN"))
    if not bin_files:
        print(f"[WARN] No ArduPilot BIN files found in: {logs_dir}")
        return []

    for bin_file in bin_files:
        print(f"[INFO] Converting ArduPilot BIN: {bin_file}")

        # pybinlog documents bin2csv as:
        #   bin2csv [-h] [-m Messages] [-o output_directory] [filepath]
        cmd = [
            "bin2csv",
            "-o",
            str(processed_dir),
            str(bin_file),
        ]

        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"bin2csv failed for {bin_file}\n"
                f"stdout:\n{result.stdout}\n\n"
                f"stderr:\n{result.stderr}"
            )

    return sorted(processed_dir.glob("*.csv"))


def process_run_directory(
    run_dir: Path,
    process_px4: bool = True,
    process_ardu: bool = True,
    skip_existing: bool = False,
) -> dict[str, list[Path]]:
    """
    Process one batch run directory.

    Parameters
    ----------
    run_dir : Path
        Path to a run directory such as ``/data/run_000``.
    process_px4 : bool, optional
        Whether to process PX4 logs.
    process_ardu : bool, optional
        Whether to process ArduPilot logs.
    skip_existing : bool, optional
        If True, skip conversion when processed CSV files already exist.

    Returns
    -------
    dict[str, list[Path]]
        Dictionary with keys:
        - ``"px4"``  : generated PX4 CSV files
        - ``"ardu"`` : generated ArduPilot CSV files
    """
    run_dir = Path(run_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    results: dict[str, list[Path]] = {"px4": [], "ardu": []}

    if process_px4:
        px4_logs_dir = run_dir / "px4_logs"
        if px4_logs_dir.exists():
            results["px4"] = convert_px4_ulog_to_csv(
                px4_logs_dir=px4_logs_dir,
                skip_existing=skip_existing,
            )
        else:
            print(f"[WARN] Missing PX4 log directory: {px4_logs_dir}")

    if process_ardu:
        ardu_logs_dir = run_dir / "ardu_logs"
        if ardu_logs_dir.exists():
            results["ardu"] = convert_ardupilot_bin_to_csv(
                ardu_logs_dir=ardu_logs_dir,
                skip_existing=skip_existing,
            )
        else:
            print(f"[WARN] Missing ArduPilot log directory: {ardu_logs_dir}")

    return results


def _iter_run_dirs(data_root: Path) -> Iterable[Path]:
    """
    Yield batch run directories under the given data root.

    A valid run directory is assumed to match:
        run_*

    Parameters
    ----------
    data_root : Path
        Root batch directory, e.g. ``/data``.

    Yields
    ------
    Path
        Run directory paths sorted lexicographically.
    """
    yield from sorted(p for p in data_root.glob("run_*") if p.is_dir())


def process_all_runs(
    data_root: Path,
    process_px4: bool = True,
    process_ardu: bool = True,
    skip_existing: bool = False,
) -> dict[str, dict[str, list[Path]]]:
    """
    Process all run directories under a batch data root.

    Parameters
    ----------
    data_root : Path
        Root directory containing ``run_XXX`` subdirectories.
    process_px4 : bool, optional
        Whether to process PX4 logs.
    process_ardu : bool, optional
        Whether to process ArduPilot logs.
    skip_existing : bool, optional
        If True, skip conversion when processed CSV files already exist.

    Returns
    -------
    dict[str, dict[str, list[Path]]]
        Nested result dictionary of the form:

        {
            "run_000": {"px4": [...], "ardu": [...]},
            "run_001": {"px4": [...], "ardu": [...]},
            ...
        }
    """
    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    results: dict[str, dict[str, list[Path]]] = {}

    run_dirs = list(_iter_run_dirs(data_root))
    if not run_dirs:
        print(f"[WARN] No run_* directories found in: {data_root}")
        return results

    for run_dir in run_dirs:
        print(f"[INFO] Processing run directory: {run_dir}")
        results[run_dir.name] = process_run_directory(
            run_dir=run_dir,
            process_px4=process_px4,
            process_ardu=process_ardu,
            skip_existing=skip_existing,
        )

    return results



def main() -> int:
    """
    CLI entry point.

    Returns
    -------
    int
        Process exit code.
    """

    # Configured parser for the conversion script
    parser = argparse.ArgumentParser(
        description="Convert PX4 ULog and ArduPilot BIN logs to CSV."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dir",
        type=Path,
        help="Process a single run directory such as /data/run_000",
    )
    group.add_argument(
        "--data-root",
        type=Path,
        help="Process all run_* directories under a data root such as /data",
    )

    parser.add_argument(
        "--px4-only",
        action="store_true",
        help="Process only PX4 logs",
    )
    parser.add_argument(
        "--ardu-only",
        action="store_true",
        help="Process only ArduPilot logs",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conversion if processed CSV files already exist",
    )
    args = parser.parse_args()

    # Resolve which log families to process.
    process_px4 = not args.ardu_only
    process_ardu = not args.px4_only

    try:
        if args.run_dir is not None:
            process_run_directory(
                run_dir=args.run_dir,
                process_px4=process_px4,
                process_ardu=process_ardu,
                skip_existing=args.skip_existing,
            )
        else:
            process_all_runs(
                data_root=args.data_root,
                process_px4=process_px4,
                process_ardu=process_ardu,
                skip_existing=args.skip_existing,
            )

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())