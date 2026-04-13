#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pyulog import ULog
from pymavlink import mavutil
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

matplotlib.use("Agg")

TARGET_HZ = 50.0
DT = 1.0 / TARGET_HZ  # 20 ms


# =====================================================================
# HELPERS
# =====================================================================
def _safe_savgol(signal, window_length=21, polyorder=2):
    """
    Apply Savitzky-Golay smoothing safely.

    If the signal is too short, return the original signal.
    """
    n = len(signal)
    if n < 5:
        return signal

    wl = min(window_length, n if n % 2 == 1 else n - 1)
    if wl < 5:
        return signal
    if wl <= polyorder:
        wl = polyorder + 2
        if wl % 2 == 0:
            wl += 1
        if wl > n:
            return signal

    return savgol_filter(signal, window_length=wl, polyorder=polyorder)


def _mask_to_ranges(mask: np.ndarray):
    """
    Convert a boolean mask into half-open ranges [start, end).
    """
    if len(mask) == 0:
        return []

    edges = np.diff(mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)

    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]

    return list(zip(starts, ends))


def _remove_short_true_runs(mask: np.ndarray, min_len: int):
    """
    Remove True runs shorter than min_len.
    """
    out = mask.copy()
    for s, e in _mask_to_ranges(mask):
        if (e - s) < min_len:
            out[s:e] = False
    return out


def _fill_short_false_gaps(mask: np.ndarray, max_gap: int):
    """
    Fill False gaps shorter than or equal to max_gap,
    excluding boundary gaps.
    """
    out = mask.copy()
    inv = ~mask
    for s, e in _mask_to_ranges(inv):
        if s == 0 or e == len(mask):
            continue
        if (e - s) <= max_gap:
            out[s:e] = True
    return out


def _make_segment(label, tt, ff, s_idx, e_idx):
    """
    Create a segment dictionary for [s_idx, e_idx).
    """
    if e_idx <= s_idx:
        return None

    safe_e = min(e_idx, len(tt))
    if safe_e <= s_idx:
        return None

    return {
        "label": label,
        "time": tt[s_idx:safe_e],
        "features": ff[s_idx:safe_e],
        "span": (tt[s_idx], tt[safe_e - 1]),
        "index_span": (s_idx, safe_e),
    }


def _empty_segment_result():
    spans = {
        "takeoff": None,
        "landing": None,
        "straight": [],
        "turn": [],
        "unknown": [],
        "straight_const": [],
        "straight_accel": [],
        "straight_decel": [],
    }
    segments = {
        "takeoff": None,
        "landing": None,
        "straight": [],
        "turn": [],
        "unknown": [],
        "straight_const": [],
        "straight_accel": [],
        "straight_decel": [],
    }
    return segments, spans


# =====================================================================
# FEATURE EXTRACTION
# =====================================================================
def extract_kinematic_features(
    t,
    x,
    y,
    z,
    vx,
    vy,
    vz,
    target_hz=50.0,
    smooth_window=21,
    poly_order=2,
):
    """
    Extract uniformly resampled kinematic features.

    Feature columns
    ---------------
    0 : altitude
    1 : heading
    2 : vertical velocity
    3 : horizontal speed
    4 : vertical acceleration
    5 : horizontal acceleration norm
    6 : vertical jerk
    7 : horizontal jerk norm
    8 : curvature
    9 : yaw rate

    Notes
    -----
    - Input timestamps are de-duplicated before interpolation.
    - Heading and yaw-rate are derived from smoothed horizontal velocity.
    - Curvature is gated by horizontal speed to avoid blow-up at low speed.
    """
    t, unique_indices = np.unique(t, return_index=True)
    x, y, z = x[unique_indices], y[unique_indices], z[unique_indices]
    vx, vy, vz = vx[unique_indices], vy[unique_indices], vz[unique_indices]

    if len(t) < 2:
        return None

    dt = 1.0 / target_hz
    t_start, t_end = t[0], t[-1]
    t_new = np.arange(t_start, t_end, dt)

    if len(t_new) < 2:
        return None

    x_new = interp1d(t, x, bounds_error=False, fill_value="extrapolate")(t_new)
    y_new = interp1d(t, y, bounds_error=False, fill_value="extrapolate")(t_new)
    z_new = interp1d(t, z, bounds_error=False, fill_value="extrapolate")(t_new)
    vx_new = interp1d(t, vx, bounds_error=False, fill_value="extrapolate")(t_new)
    vy_new = interp1d(t, vy, bounds_error=False, fill_value="extrapolate")(t_new)
    vz_new = interp1d(t, vz, bounds_error=False, fill_value="extrapolate")(t_new)

    z_smooth = _safe_savgol(z_new, window_length=smooth_window, polyorder=poly_order)
    vx_smooth = _safe_savgol(vx_new, window_length=smooth_window, polyorder=poly_order)
    vy_smooth = _safe_savgol(vy_new, window_length=smooth_window, polyorder=poly_order)
    vz_smooth = _safe_savgol(vz_new, window_length=smooth_window, polyorder=poly_order)

    altitude = -z_smooth
    v_alt = -vz_smooth

    v_xy = np.vstack((vx_smooth, vy_smooth)).T
    speed_xy = np.linalg.norm(v_xy, axis=1)

    ax = np.gradient(vx_smooth, dt)
    ay = np.gradient(vy_smooth, dt)
    az = np.gradient(vz_smooth, dt)

    a_alt = -az
    a_xy = np.vstack((ax, ay)).T
    acc_norm_xy = np.linalg.norm(a_xy, axis=1)

    ax_s = _safe_savgol(ax, window_length=smooth_window, polyorder=poly_order)
    ay_s = _safe_savgol(ay, window_length=smooth_window, polyorder=poly_order)
    az_s = _safe_savgol(az, window_length=smooth_window, polyorder=poly_order)

    jx = np.gradient(ax_s, dt)
    jy = np.gradient(ay_s, dt)
    jz = np.gradient(az_s, dt)

    j_alt = -jz
    j_xy = np.vstack((jx, jy)).T
    jerk_norm_xy = np.linalg.norm(j_xy, axis=1)

    heading = np.unwrap(np.arctan2(vy_smooth, vx_smooth))
    raw_yaw_rate = np.gradient(heading, dt)
    yaw_rate = _safe_savgol(raw_yaw_rate, window_length=smooth_window, polyorder=poly_order)

    v_vec_3d = np.vstack((vx_smooth, vy_smooth, vz_smooth)).T
    a_vec_3d = np.vstack((ax, ay, az)).T
    cross_va = np.cross(v_vec_3d, a_vec_3d)
    cross_mag = np.linalg.norm(cross_va, axis=1)

    speed_3d = np.linalg.norm(v_vec_3d, axis=1)
    raw_curvature = cross_mag / np.maximum(speed_3d**3, 1e-3)

    # Additional gating for low-speed robustness
    raw_curvature[speed_xy < 0.8] = 0.0
    curvature = _safe_savgol(raw_curvature, window_length=smooth_window, polyorder=poly_order)

    features = np.vstack(
        (
            altitude,
            heading,
            v_alt,
            speed_xy,
            a_alt,
            acc_norm_xy,
            j_alt,
            jerk_norm_xy,
            curvature,
            yaw_rate,
        )
    ).T

    return t_new, features


# =====================================================================
# SEGMENTATION V2
# =====================================================================
def extract_flight_segments_v2(
    t,
    features,
    dt=0.02,
    altitude_margin=0.1,
    vz_small=0.12,
    az_small=0.20,
    min_moving_speed=0.8,
    min_valid_speed_for_heading=1.0,
    yaw_rate_on=0.18,
    yaw_rate_off=0.08,
    curvature_on=0.03,
    curvature_off=0.015,
    min_turn_duration=0.6,
    min_straight_duration=1.0,
    straight_yaw_rate_max=0.08,
    straight_curvature_max=0.015,
    long_acc_threshold=0.4,
    min_accel_duration=0.4,
    max_merge_gap=0.25,
    min_unknown_duration=0.3,
):
    """
    Improved flight segmentation with straight subtypes.

    Primary classes
    ---------------
    - takeoff
    - landing
    - straight
    - turn
    - unknown

    Straight subtypes
    -----------------
    - straight_const
    - straight_accel
    - straight_decel

    Design choice
    -------------
    Turn is treated as a single semantic maneuver even if acceleration
    or deceleration occurs within it. Straight segments are further split
    into longitudinal speed-change subtypes because those are often useful
    for autopilot discrimination.
    """
    segments, spans = _empty_segment_result()

    if features is None or len(features) < 2 or len(t) != len(features):
        return segments, spans

    N = len(features)

    altitude = features[:, 0]
    v_alt = features[:, 2]
    speed_xy = features[:, 3]
    a_alt = features[:, 4]
    curvature = features[:, 8]
    yaw_rate = features[:, 9]

    # --------------------------------------------------------------
    # Step 1. Crop takeoff / landing using altitude + vertical settling
    # --------------------------------------------------------------
    alt_95 = np.percentile(altitude, 95)
    target_alt = alt_95 - altitude_margin

    flight_start_idx = 0
    state = 0
    for i in range(N):
        if state == 0 and altitude[i] >= target_alt:
            state = 1
        elif state == 1 and abs(v_alt[i]) <= vz_small:
            state = 2
        elif state == 2 and abs(a_alt[i]) <= az_small:
            flight_start_idx = i
            break

    flight_end_idx = N
    state = 0
    for i in range(N - 1, flight_start_idx, -1):
        if state == 0 and altitude[i] >= target_alt:
            state = 1
        elif state == 1 and abs(v_alt[i]) <= vz_small:
            state = 2
        elif state == 2 and abs(a_alt[i]) <= az_small:
            flight_end_idx = i
            break

    if flight_start_idx > 0:
        seg = _make_segment("takeoff", t, features, 0, flight_start_idx)
        if seg is not None:
            segments["takeoff"] = seg
            spans["takeoff"] = seg["span"]

    if flight_end_idx < N - 1:
        seg = _make_segment("landing", t, features, flight_end_idx, N)
        if seg is not None:
            segments["landing"] = seg
            spans["landing"] = seg["span"]

    flight_t = t[flight_start_idx:flight_end_idx]
    flight_features = features[flight_start_idx:flight_end_idx]

    if len(flight_features) < 2:
        return segments, spans

    speed_xy_f = flight_features[:, 3]
    curvature_f = flight_features[:, 8]
    yaw_rate_f = flight_features[:, 9]

    M = len(flight_features)

    # longitudinal acceleration in straight analysis
    speed_xy_smooth = _safe_savgol(speed_xy_f, window_length=21, polyorder=2)
    a_long = np.gradient(speed_xy_smooth, dt)

    is_moving = speed_xy_f >= min_moving_speed
    has_valid_heading = speed_xy_f >= min_valid_speed_for_heading

    # --------------------------------------------------------------
    # Step 2. Explicit turn detection with hysteresis
    # --------------------------------------------------------------
    raw_turn_on = (
        has_valid_heading
        & (
            (np.abs(yaw_rate_f) >= yaw_rate_on)
            | (curvature_f >= curvature_on)
        )
    )

    raw_turn_off = (
        has_valid_heading
        & (
            (np.abs(yaw_rate_f) >= yaw_rate_off)
            | (curvature_f >= curvature_off)
        )
    )

    is_turn = np.zeros(M, dtype=bool)
    active = False
    for i in range(M):
        if not active:
            if raw_turn_on[i]:
                active = True
                is_turn[i] = True
        else:
            if raw_turn_off[i]:
                is_turn[i] = True
            else:
                active = False

    min_turn_len = max(1, int(round(min_turn_duration / dt)))
    merge_gap_len = max(0, int(round(max_merge_gap / dt)))

    is_turn = _fill_short_false_gaps(is_turn, merge_gap_len)
    is_turn = _remove_short_true_runs(is_turn, min_turn_len)

    # --------------------------------------------------------------
    # Step 3. Straight detection
    # --------------------------------------------------------------
    raw_straight = (
        is_moving
        & (~is_turn)
        & (np.abs(yaw_rate_f) <= straight_yaw_rate_max)
        & (curvature_f <= straight_curvature_max)
    )

    min_straight_len = max(1, int(round(min_straight_duration / dt)))
    is_straight = _fill_short_false_gaps(raw_straight, merge_gap_len)
    is_straight = _remove_short_true_runs(is_straight, min_straight_len)

    # --------------------------------------------------------------
    # Step 4. Unknown region
    # --------------------------------------------------------------
    is_unknown = ~(is_turn | is_straight)

    min_unknown_len = max(1, int(round(min_unknown_duration / dt)))
    is_unknown = _remove_short_true_runs(is_unknown, min_unknown_len)

    # normalize priority
    is_straight = is_straight & (~is_turn)
    is_unknown = ~(is_turn | is_straight)

    # --------------------------------------------------------------
    # Step 5. Export primary segments
    # --------------------------------------------------------------
    for s, e in _mask_to_ranges(is_turn):
        seg = _make_segment("turn", flight_t, flight_features, s, e)
        if seg is not None:
            segments["turn"].append(seg)
            spans["turn"].append(seg["span"])

    for s, e in _mask_to_ranges(is_straight):
        seg = _make_segment("straight", flight_t, flight_features, s, e)
        if seg is not None:
            segments["straight"].append(seg)
            spans["straight"].append(seg["span"])

    for s, e in _mask_to_ranges(is_unknown):
        seg = _make_segment("unknown", flight_t, flight_features, s, e)
        if seg is not None:
            segments["unknown"].append(seg)
            spans["unknown"].append(seg["span"])

    # --------------------------------------------------------------
    # Step 6. Straight sub-segmentation
    # --------------------------------------------------------------
    min_acc_len = max(1, int(round(min_accel_duration / dt)))

    for s, e in _mask_to_ranges(is_straight):
        local_a_long = a_long[s:e]

        local_accel = local_a_long >= long_acc_threshold
        local_decel = local_a_long <= -long_acc_threshold

        local_accel = _fill_short_false_gaps(local_accel, merge_gap_len)
        local_accel = _remove_short_true_runs(local_accel, min_acc_len)

        local_decel = _fill_short_false_gaps(local_decel, merge_gap_len)
        local_decel = _remove_short_true_runs(local_decel, min_acc_len)

        local_const = ~(local_accel | local_decel)

        for ls, le in _mask_to_ranges(local_accel):
            gs, ge = s + ls, s + le
            seg = _make_segment("straight_accel", flight_t, flight_features, gs, ge)
            if seg is not None:
                segments["straight_accel"].append(seg)
                spans["straight_accel"].append(seg["span"])

        for ls, le in _mask_to_ranges(local_decel):
            gs, ge = s + ls, s + le
            seg = _make_segment("straight_decel", flight_t, flight_features, gs, ge)
            if seg is not None:
                segments["straight_decel"].append(seg)
                spans["straight_decel"].append(seg["span"])

        for ls, le in _mask_to_ranges(local_const):
            gs, ge = s + ls, s + le
            seg = _make_segment("straight_const", flight_t, flight_features, gs, ge)
            if seg is not None:
                segments["straight_const"].append(seg)
                spans["straight_const"].append(seg["span"])

    return segments, spans


# =====================================================================
# DATA PROCESSING
# =====================================================================
def process_px4_flight_data(ulog_path):
    try:
        ulog = ULog(ulog_path)
        loc_data = ulog.get_dataset("vehicle_local_position").data

        t_loc = loc_data["timestamp"] / 1e6
        x, y, z = loc_data["x"], loc_data["y"], loc_data["z"]
        vx, vy, vz = loc_data["vx"], loc_data["vy"], loc_data["vz"]

        extracted = extract_kinematic_features(t_loc, x, y, z, vx, vy, vz, target_hz=TARGET_HZ)
        if extracted is None:
            return None, None, None, None, None, None

        t_full, feat_full = extracted
        segments, spans = extract_flight_segments_v2(t_full, feat_full, dt=DT)

        return x, y, t_full, feat_full, segments, spans

    except Exception as e:
        print(f"[PX4 Extract Error] {ulog_path}: {e}")
        return None, None, None, None, None, None


def process_ardu_flight_data(bin_path):
    try:
        mlog = mavutil.mavlink_connection(bin_path)
        t_loc, x, y, z, vx, vy, vz = [], [], [], [], [], [], []

        while True:
            msg = mlog.recv_match(type=["XKF1", "NKF1"], blocking=False)
            if not msg:
                break
            t_loc.append(msg.TimeUS / 1e6)
            x.append(msg.PN)
            y.append(msg.PE)
            z.append(msg.PD)
            vx.append(msg.VN)
            vy.append(msg.VE)
            vz.append(msg.VD)

        if len(x) < 50:
            return None, None, None, None, None, None

        extracted = extract_kinematic_features(
            np.array(t_loc),
            np.array(x),
            np.array(y),
            np.array(z),
            np.array(vx),
            np.array(vy),
            np.array(vz),
            target_hz=TARGET_HZ,
        )
        if extracted is None:
            return None, None, None, None, None, None

        t_full, feat_full = extracted
        segments, spans = extract_flight_segments_v2(t_full, feat_full, dt=DT)

        return x, y, t_full, feat_full, segments, spans

    except Exception as e:
        print(f"[ArduPilot Extract Error] {bin_path}: {e}")
        return None, None, None, None, None, None


def process_rosbag_flight_data(csv_path):
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        t_loc = data["bag_time_ns"] / 1e9
        x, y, z = data["x"], data["y"], data["z"]
        vx, vy, vz = data["vx"], data["vy"], data["vz"]

        extracted = extract_kinematic_features(t_loc, x, y, z, vx, vy, vz, target_hz=TARGET_HZ)
        if extracted is None:
            return None, None, None, None, None, None

        t_full, feat_full = extracted
        segments, spans = extract_flight_segments_v2(t_full, feat_full, dt=DT)

        return x, y, t_full, feat_full, segments, spans

    except Exception as e:
        print(f"[ROS Bag Extract Error] {csv_path}: {e}")
        return None, None, None, None, None, None


# =====================================================================
# PLOTS
# =====================================================================
def plot_combined_xy_trajectory(
    x_px4,
    y_px4,
    x_ardu,
    y_ardu,
    title="Combined X-Y Trajectory",
    save_path="xy_trajectory.png",
):
    if (x_px4 is None or len(x_px4) == 0) and (x_ardu is None or len(x_ardu) == 0):
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    if x_px4 is not None and len(x_px4) > 0:
        ax.plot(y_px4, x_px4, color="tab:green", linewidth=1.5, alpha=0.8, label="PX4 Path")
        ax.plot(y_px4[0], x_px4[0], marker="o", color="darkgreen", markersize=6, label="PX4 Start")
        ax.plot(y_px4[-1], x_px4[-1], marker="X", color="darkgreen", markersize=6, label="PX4 End")

    if x_ardu is not None and len(x_ardu) > 0:
        ax.plot(y_ardu, x_ardu, color="tab:orange", linewidth=1.5, alpha=0.8, label="ArduPilot Path")
        ax.plot(y_ardu[0], x_ardu[0], marker="o", color="darkorange", markersize=6, label="ArduPilot Start")
        ax.plot(y_ardu[-1], x_ardu[-1], marker="X", color="darkorange", markersize=6, label="ArduPilot End")

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("East (Y) [m]", fontsize=12)
    ax.set_ylabel("North (X) [m]", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_full_trajectory_with_spans(t, features, spans, title, save_path, line_color):
    if features is None or len(features) == 0:
        return

    feature_names = [
        "Altitude (m)",
        "Heading (rad)",
        "Z-Axis Velocity (m/s)",
        "XY-Plane Speed (m/s)",
        "Z-Axis Acceleration (m/s²)",
        "XY-Plane Accel Norm (m/s²)",
        "Z-Axis Jerk (m/s³)",
        "XY-Plane Jerk Norm (m/s³)",
        "Curvature (1/m)",
        "Yaw Rate (rad/s)",
    ]

    fig, axes = plt.subplots(5, 2, figsize=(16, 18))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    axes_flat = axes.flatten()

    span_colors = {
        "takeoff": "orange",
        "landing": "mediumpurple",
        "straight_const": "mediumseagreen",
        "straight_accel": "dodgerblue",
        "straight_decel": "deepskyblue",
        "turn": "crimson",
        "unknown": "gray",
    }

    for i in range(10):
        ax = axes_flat[i]
        ax.plot(t, features[:, i], color=line_color, linewidth=1.5, zorder=2)
        ax.set_title(feature_names[i], fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5, zorder=1)

        if i == 6:
            ax.set_ylim(np.percentile(features[:, i], 2), np.percentile(features[:, i], 98))
        elif i == 7:
            ax.set_ylim(-1, np.percentile(features[:, i], 98))
        elif i == 8:
            ax.set_ylim(-0.1, np.percentile(features[:, i], 98))

        if spans:
            if spans.get("takeoff"):
                ax.axvspan(*spans["takeoff"], color=span_colors["takeoff"], alpha=0.20, zorder=0)
            if spans.get("landing"):
                ax.axvspan(*spans["landing"], color=span_colors["landing"], alpha=0.20, zorder=0)

            for s, e in spans.get("straight_const", []):
                ax.axvspan(s, e, color=span_colors["straight_const"], alpha=0.18, zorder=0)

            for s, e in spans.get("straight_accel", []):
                ax.axvspan(s, e, color=span_colors["straight_accel"], alpha=0.22, zorder=0)

            for s, e in spans.get("straight_decel", []):
                ax.axvspan(s, e, color=span_colors["straight_decel"], alpha=0.22, zorder=0)

            for s, e in spans.get("turn", []):
                ax.axvspan(s, e, color=span_colors["turn"], alpha=0.30, zorder=0)

            for s, e in spans.get("unknown", []):
                ax.axvspan(s, e, color=span_colors["unknown"], alpha=0.12, zorder=0)

    legend_patches = [
        mpatches.Patch(color=span_colors["takeoff"], alpha=0.20, label="Takeoff"),
        mpatches.Patch(color=span_colors["straight_const"], alpha=0.18, label="Straight Const"),
        mpatches.Patch(color=span_colors["straight_accel"], alpha=0.22, label="Straight Accel"),
        mpatches.Patch(color=span_colors["straight_decel"], alpha=0.22, label="Straight Decel"),
        mpatches.Patch(color=span_colors["turn"], alpha=0.30, label="Turn"),
        mpatches.Patch(color=span_colors["unknown"], alpha=0.12, label="Unknown"),
        mpatches.Patch(color=span_colors["landing"], alpha=0.20, label="Landing"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(0.98, 0.98), ncol=4, fontsize=11)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_turn_segment_features(t, features, title, save_path, line_color):
    if features is None or len(features) == 0:
        return

    feature_names = [
        "Altitude (m)",
        "Heading (rad)",
        "Z-Axis Velocity (m/s)",
        "XY-Plane Speed (m/s)",
        "Z-Axis Acceleration (m/s²)",
        "XY-Plane Accel Norm (m/s²)",
        "Z-Axis Jerk (m/s³)",
        "XY-Plane Jerk Norm (m/s³)",
        "Curvature (1/m)",
        "Yaw Rate (rad/s)",
    ]

    fig, axes = plt.subplots(5, 2, figsize=(16, 18))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    axes_flat = axes.flatten()

    for i in range(10):
        ax = axes_flat[i]
        ax.plot(t, features[:, i], color=line_color, linewidth=2.0)
        ax.set_title(feature_names[i], fontsize=12, fontweight="bold")
        ax.set_xlabel("Absolute Time (s)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)

        if i == 6:
            ax.set_ylim(np.percentile(features[:, i], 2), np.percentile(features[:, i], 98))
        elif i == 7:
            ax.set_ylim(-1, np.percentile(features[:, i], 98))
        elif i == 8:
            ax.set_ylim(-0.1, np.percentile(features[:, i], 98))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    BASE_DATA_DIR = Path("data")
    print("[Info] Starting Combined Pipeline: XY / Full Highlights / Segments...\n")

    for i in range(100):
        run_folder = f"run_{i:03d}"
        run_dir = BASE_DATA_DIR / run_folder
        if not run_dir.exists():
            continue

        px4_dir = run_dir / "px4_logs" / "raw"
        ardu_dir = run_dir / "ardu_logs" / "raw" / "logs"

        x_px4, y_px4, t_px4, feat_px4, segments_px4, spans_px4 = (None,) * 6
        x_ardu, y_ardu, t_ardu, feat_ardu, segments_ardu, spans_ardu = (None,) * 6

        if px4_dir.exists():
            for file in os.listdir(px4_dir):
                if file.lower().endswith(".ulg"):
                    px4_result = process_px4_flight_data(str(px4_dir / file))
                    x_px4, y_px4, t_px4, feat_px4, segments_px4, spans_px4 = px4_result
                    break

        if ardu_dir.exists():
            for file in os.listdir(ardu_dir):
                if file.lower().endswith(".bin"):
                    ardu_result = process_ardu_flight_data(str(ardu_dir / file))
                    x_ardu, y_ardu, t_ardu, feat_ardu, segments_ardu, spans_ardu = ardu_result
                    break

        if (x_px4 is not None) or (x_ardu is not None):
            plot_combined_xy_trajectory(
                x_px4,
                y_px4,
                x_ardu,
                y_ardu,
                title=f"Combined X-Y Trajectory ({run_folder})",
                save_path=str(run_dir / f"trajectory_xy_combined_{run_folder}.png"),
            )

        if feat_px4 is not None and len(feat_px4) > 0:
            print(f"[{run_folder}] Generating PX4 trajectory plot...")
            plot_full_trajectory_with_spans(
                t=t_px4,
                features=feat_px4,
                spans=spans_px4,
                title=f"Trajectory [Segment Check]: PX4 ({run_folder})",
                save_path=str(run_dir / f"features_px4_seg_check_{run_folder}.png"),
                line_color="tab:green",
            )

            if segments_px4 and segments_px4["turn"]:
                first_turn = segments_px4["turn"][0]
                plot_turn_segment_features(
                    t=first_turn["time"],
                    features=first_turn["features"],
                    title=f"Trajectory [Isolated Turn]: PX4 ({run_folder})",
                    save_path=str(run_dir / f"features_px4_turn_seg_{run_folder}.png"),
                    line_color="tab:green",
                )

        if feat_ardu is not None and len(feat_ardu) > 0:
            print(f"[{run_folder}] Generating ArduPilot trajectory plot...")
            plot_full_trajectory_with_spans(
                t=t_ardu,
                features=feat_ardu,
                spans=spans_ardu,
                title=f"Trajectory [Segment Check]: ArduPilot ({run_folder})",
                save_path=str(run_dir / f"features_ardupilot_seg_check_{run_folder}.png"),
                line_color="tab:orange",
            )

            if segments_ardu and segments_ardu["turn"]:
                first_turn = segments_ardu["turn"][0]
                plot_turn_segment_features(
                    t=first_turn["time"],
                    features=first_turn["features"],
                    title=f"Trajectory [Isolated Turn]: ArduPilot ({run_folder})",
                    save_path=str(run_dir / f"features_ardupilot_turn_seg_{run_folder}.png"),
                    line_color="tab:orange",
                )

    print("\n[Info] All combined XY and segmentation plots generated successfully!")