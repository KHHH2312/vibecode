"""
Biohub - Cell Tracking During Development  |  CPU-only full-inference pipeline.

Pipeline (recomputed from the image volumes on every run -- this is NOT a
postprocess-only kernel; it never rewrites a precomputed CSV):

    detect        3D nuclei per timepoint (adaptive-threshold local maxima)
      -> link     frame-to-frame association (KD-tree gated, motion-aware,
                  ascending-distance mutual matching -- stronger than greedy NN)
      -> gap-close recover missed detections across small temporal gaps,
                  optionally interpolating the skipped frames (gap recovery)
      -> divide   parent -> two spatially-symmetric, velocity-consistent daughters
      -> assemble tracks + parent links, drop spurious short tracks
      -> write     submission.csv, columns/units matched to sample_submission.csv
                  discovered at runtime (primary defense against FORMAT_FAIL)

Targets Kaggle: CPU only, internet off, competition time limits. Reads OME-Zarr
v3 volumes; picks a downsampled pyramid level to stay within the CPU budget and
rescales detections back to full-resolution voxel coordinates.

See README.md for design notes, first-run checks, and honest score caveats.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration -- every tunable lives here.  Distances are in micrometres (um)
# because the competition metric matches edges at a 7 um tolerance.
# --------------------------------------------------------------------------- #
CONFIG = {
    # -- runtime / pyramid ------------------------------------------------- #
    "target_max_voxels": 48_000_000,  # pick the smallest pyramid level whose
    #                                   per-frame volume is <= this (CPU budget)
    "max_timepoints": None,           # None = all frames; int caps for debugging
    # -- detection --------------------------------------------------------- #
    "cell_radius_um": 5.0,            # approx nuclear radius -> smoothing sigma
    "detect_min_distance_um": 6.0,    # minimum separation between detections
    "detect_threshold_mode": "mad",   # 'mad' | 'otsu' | 'percentile' | 'mean_std'
    "detect_mad_k": 6.0,              # threshold = median + k*1.4826*MAD (mad mode)
    "detect_mean_std_k": 3.0,         # threshold = mean + k*std (mean_std mode)
    "detect_percentile": 99.0,        # foreground percentile (percentile mode)
    "max_detections_per_frame": 80_000,  # safety cap (keeps brightest)
    # -- linking ----------------------------------------------------------- #
    "max_link_dist_um": 12.0,         # max plausible frame-to-frame motion
    "use_motion_prior": True,         # subtract running median flow before match
    # -- gap closing ------------------------------------------------------- #
    "max_gap": 3,                     # bridge tracks across up to this many frames
    "gap_dist_um": 8.0,               # per-frame radius; scaled by the gap length
    "interpolate_gaps": True,         # insert nodes on skipped frames (recall)
    # -- divisions --------------------------------------------------------- #
    "enable_divisions": True,
    "sister_radius_um": 12.0,         # max parent->daughter distance
    "sister_pair_max_um": 18.0,       # max daughter<->daughter distance
    "division_symmetry_tol_um": 7.0,  # parent must sit near the daughters' midpoint
    # -- assembly / output ------------------------------------------------- #
    "min_track_len": 3,               # drop isolated tracks shorter than this
    "coord_mode": "voxel",            # 'voxel' (full-res indices) | 'um'
    "verbose": True,
}


def log(msg, cfg=CONFIG):
    if cfg.get("verbose", True):
        print(f"[biohub] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Dependency import (zarr may be missing on an internet-off Kaggle image)
# --------------------------------------------------------------------------- #
def import_zarr():
    """Import zarr, falling back to offline wheels attached as a Kaggle dataset."""
    try:
        import zarr
        return zarr
    except ImportError:
        pass
    # Look for any attached dataset that ships zarr wheels and install offline.
    patterns = ["*zarr*", "*wheel*", "*whl*", "*offline*", "*deps*", "*pip*"]
    seen = []
    for pat in patterns:
        for d in sorted(glob.glob(os.path.join("/kaggle/input", pat))):
            if d in seen or not os.path.isdir(d):
                continue
            seen.append(d)
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--no-index",
                     "--find-links", d, "zarr"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                import zarr
                return zarr
            except Exception:
                continue
    raise ImportError(
        "Could not import zarr and no offline wheel dataset was found. Attach a "
        "Kaggle dataset containing zarr (+ numcodecs) wheels; see README.md."
    )


# --------------------------------------------------------------------------- #
# OME-Zarr reading
# --------------------------------------------------------------------------- #
class Dataset:
    """A single test volume: lazy per-timepoint access + physical spacing."""

    def __init__(self, name, arr, axis_idx, T, level_scale_um_zyx, full_over_level_zyx):
        self.name = name
        self._arr = arr                      # zarr array at the chosen level
        self._ax = axis_idx                  # {'t','c','z','y','x'} -> axis or None
        self.T = T                           # number of timepoints
        self.spacing_um = level_scale_um_zyx  # um per voxel at the chosen level
        self.vox_to_full = full_over_level_zyx  # multiply level-voxel -> full-res

    def frame(self, t):
        """Return the (Z, Y, X) volume at timepoint t as a float32 array."""
        arr = self._arr
        idx = [slice(None)] * arr.ndim
        if self._ax["t"] is not None:
            idx[self._ax["t"]] = t
        vol = np.asarray(arr[tuple(idx)], dtype=np.float32)
        # Reduce any leftover channel axis by max-projection.
        if self._ax["c"] is not None:
            c_axis = self._ax["c"]
            if self._ax["t"] is not None and self._ax["t"] < c_axis:
                c_axis -= 1
            vol = vol.max(axis=c_axis)
        # Ensure 3D (Z, Y, X); promote 2D frames to a single z-slice.
        if vol.ndim == 2:
            vol = vol[None, ...]
        return vol


def _get_multiscales(attrs):
    """OME-NGFF 0.4 stores 'multiscales' at the top; 0.5 nests it under 'ome'."""
    d = dict(attrs)
    if "multiscales" in d:
        return d["multiscales"]
    if "ome" in d and isinstance(d["ome"], dict) and "multiscales" in d["ome"]:
        return d["ome"]["multiscales"]
    return None


def _axis_index(axes_names):
    """Map t/c/z/y/x (case-insensitive) to their positions in the array."""
    idx = {"t": None, "c": None, "z": None, "y": None, "x": None}
    for i, nm in enumerate(axes_names):
        k = str(nm).lower()[:1]
        if k in idx:
            idx[k] = i
    return idx


def open_dataset(zarr, root_path, name, cfg=CONFIG):
    """Open an OME-Zarr group (or bare array) and choose a pyramid level."""
    node = zarr.open_group(root_path, mode="r")
    ms = _get_multiscales(node.attrs)

    if ms:
        ms0 = ms[0]
        axes = ms0.get("axes", [])
        axes_names = [a.get("name", a) if isinstance(a, dict) else a for a in axes]
        units = [a.get("unit", "") if isinstance(a, dict) else "" for a in axes]
        levels = ms0["datasets"]  # ordered fine -> coarse

        def scale_of(ds):
            for tr in ds.get("coordinateTransformations", []):
                if tr.get("type") == "scale":
                    return list(tr["scale"])
            return [1.0] * len(axes_names)

        scale0 = scale_of(levels[0])
        # Pick the finest level small enough to fit the per-frame voxel budget.
        chosen, chosen_scale = levels[0], scale0
        for ds in levels:
            arr = node[ds["path"]]
            ax = _axis_index(axes_names)
            spatial = [arr.shape[i] for i in (ax["z"], ax["y"], ax["x"]) if i is not None]
            nvox = int(np.prod(spatial)) if spatial else 0
            if nvox <= cfg["target_max_voxels"]:
                chosen, chosen_scale = ds, scale_of(ds)
                break
            chosen, chosen_scale = ds, scale_of(ds)  # fall back to coarsest
        arr = node[chosen["path"]]
        ax = _axis_index(axes_names)
    else:
        # Bare array or plain group: assume (T, Z, Y, X) or (Z, Y, X).
        arr = node if hasattr(node, "shape") else node[sorted(node.array_keys())[0]]
        if arr.ndim >= 4:
            axes_names, units = ["t", "z", "y", "x"], ["", "", "", ""]
        else:
            axes_names, units = ["z", "y", "x"], ["", "", ""]
        ax = _axis_index(axes_names)
        chosen_scale = [1.0] * arr.ndim
        scale0 = chosen_scale

    def um_scale(axis_key):
        i = ax[axis_key]
        if i is None:
            return 1.0
        unit = str(units[i]).lower() if i < len(units) else ""
        val = float(chosen_scale[i]) if i < len(chosen_scale) else 1.0
        # OME units are typically micrometre already; convert nm/mm if declared.
        if "nano" in unit:
            val *= 1e-3
        elif "milli" in unit:
            val *= 1e3
        return val

    spacing = np.array([um_scale("z"), um_scale("y"), um_scale("x")], dtype=float)
    spacing[spacing <= 0] = 1.0
    # Ratio of chosen-level scale to full-res (level 0) scale = downsample factor.
    full_over_level = np.array(
        [float(chosen_scale[ax[a]]) / float(scale0[ax[a]]) if ax[a] is not None
         and ax[a] < len(scale0) and scale0[ax[a]] else 1.0 for a in ("z", "y", "x")],
        dtype=float,
    )
    T = int(arr.shape[ax["t"]]) if ax["t"] is not None else 1
    if cfg.get("max_timepoints"):
        T = min(T, int(cfg["max_timepoints"]))
    log(f"dataset '{name}': level shape={tuple(arr.shape)} T={T} "
        f"spacing_um(zyx)={np.round(spacing, 3).tolist()} "
        f"downsample(zyx)={np.round(full_over_level, 2).tolist()}", cfg)
    return Dataset(name, arr, ax, T, spacing, full_over_level)


def discover_datasets(zarr, input_root, cfg=CONFIG):
    """Find OME-Zarr test volumes under the Kaggle input tree."""
    roots = []

    def looks_like_group(path):
        return (os.path.exists(os.path.join(path, "zarr.json"))
                or os.path.exists(os.path.join(path, ".zgroup")))

    # Prefer explicit *.zarr stores; else any zarr group under a 'test'-ish dir.
    cand = sorted(glob.glob(os.path.join(input_root, "**", "*.zarr"), recursive=True))
    if not cand:
        markers = (glob.glob(os.path.join(input_root, "**", "zarr.json"), recursive=True)
                   + glob.glob(os.path.join(input_root, "**", ".zgroup"), recursive=True))
        cand = sorted({os.path.dirname(m) for m in markers})
    # Keep only outermost groups (drop nested arrays inside a parent group).
    cand = [c for c in cand if looks_like_group(c)]
    cand.sort(key=len)
    for c in cand:
        if any(c != r and c.startswith(r + os.sep) for r in roots):
            continue  # nested inside an already-selected group
        roots.append(c)

    datasets = []
    for r in roots:
        name = os.path.splitext(os.path.basename(r.rstrip("/")))[0]
        try:
            ds = open_dataset(zarr, r, name, cfg)
            if ds.T >= 1:
                datasets.append(ds)
        except Exception as exc:  # noqa: BLE001 - skip unreadable groups
            log(f"skipping '{r}': {exc}", cfg)
    log(f"discovered {len(datasets)} test dataset(s)", cfg)
    return datasets


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def _threshold(vol, cfg):
    """Adaptive foreground threshold on the smoothed volume.

    'mad' is the default: median + k*(1.4826*MAD) is a background-relative cut
    that stays robust when the foreground fraction is small (bright nuclei on a
    dark background), unlike mean+std which the foreground itself inflates.
    """
    mode = cfg["detect_threshold_mode"]
    if mode == "otsu":
        try:
            from skimage.filters import threshold_otsu
            return float(threshold_otsu(vol))
        except Exception:
            mode = "mad"
    if mode == "percentile":
        return float(np.percentile(vol, cfg["detect_percentile"]))
    if mode == "mean_std":
        return float(vol.mean() + cfg["detect_mean_std_k"] * vol.std())
    med = float(np.median(vol))
    mad = float(np.median(np.abs(vol - med))) * 1.4826
    return med + cfg["detect_mad_k"] * mad


def detect_frame(vol_zyx, spacing_um_zyx, cfg):
    """Return (K, 3) voxel coordinates (z, y, x) of detected nuclei."""
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max

    v = np.asarray(vol_zyx, dtype=np.float32)
    lo, hi = np.percentile(v, [1.0, 99.9])
    if hi <= lo:
        return np.empty((0, 3), dtype=float)
    v = np.clip((v - lo) / (hi - lo), 0.0, 1.0)

    # Light smoothing (~radius/3) denoises without flattening the blob peaks.
    sigma_vox = [max(0.5, (cfg["cell_radius_um"] / 3.0) / s) for s in spacing_um_zyx]
    vs = ndi.gaussian_filter(v, sigma=sigma_vox)

    thr = _threshold(vs, cfg)
    finest = float(min(spacing_um_zyx))
    min_dist_vox = max(1, int(round(cfg["detect_min_distance_um"] / finest)))
    coords = peak_local_max(
        vs, min_distance=min_dist_vox, threshold_abs=thr, exclude_border=False,
    )
    if len(coords) > cfg["max_detections_per_frame"]:
        vals = vs[tuple(coords.T)]
        keep = np.argsort(vals)[::-1][: cfg["max_detections_per_frame"]]
        coords = coords[keep]
    return coords.astype(float)


def detect_all(ds, cfg):
    """Detect every timepoint; return per-frame um and full-res voxel positions."""
    frames = []
    t0 = time.time()
    for t in range(ds.T):
        vox = detect_frame(ds.frame(t), ds.spacing_um, cfg)   # (K,3) level voxels
        um = vox * ds.spacing_um[None, :]                      # physical um
        full_vox = vox * ds.vox_to_full[None, :]               # full-res voxels
        frames.append({"um": um, "vox": full_vox})
    total = sum(len(f["um"]) for f in frames)
    log(f"  detected {total} nuclei over {ds.T} frames in {time.time() - t0:.1f}s", cfg)
    return frames


# --------------------------------------------------------------------------- #
# Track graph
# --------------------------------------------------------------------------- #
class TrackGraph:
    def __init__(self):
        self.t = []          # timepoint per node
        self.um = []         # (z,y,x) um per node
        self.vox = []        # (z,y,x) full-res voxel per node
        self.parent = []     # parent node id or -1
        self.children = []   # list of child node ids
        self.interp = []     # True for interpolated (gap-filled) nodes

    def add_node(self, t, um, vox, interp=False):
        gid = len(self.t)
        self.t.append(int(t))
        self.um.append(np.asarray(um, dtype=float))
        self.vox.append(np.asarray(vox, dtype=float))
        self.parent.append(-1)
        self.children.append([])
        self.interp.append(interp)
        return gid

    def add_edge(self, a, b):
        """Directed edge a (earlier) -> b (later)."""
        self.parent[b] = a
        self.children[a].append(b)

    @property
    def n(self):
        return len(self.t)


def _match_frames(pts_a, pts_b, radius, flow=None):
    """Ascending-distance mutual matching of a->b within `radius` (um).

    Returns (matches (M,2) as [i,j], estimated median flow (3,)).
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return np.empty((0, 2), dtype=int), np.zeros(3)
    from scipy.spatial import cKDTree

    a = pts_a + (flow if flow is not None else 0.0)
    tree = cKDTree(pts_b)
    k = min(5, len(pts_b))
    dd, jj = tree.query(a, k=k, distance_upper_bound=radius)
    dd = np.atleast_2d(dd.reshape(len(a), k))
    jj = np.atleast_2d(jj.reshape(len(a), k))

    cand = []
    nb = len(pts_b)
    for i in range(len(a)):
        for c in range(k):
            d, j = dd[i, c], jj[i, c]
            if np.isfinite(d) and j < nb:
                cand.append((d, i, int(j)))
    cand.sort()

    used_a, used_b, matches = set(), set(), []
    for _, i, j in cand:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j))
    matches = np.asarray(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)
    if len(matches):
        flow_new = np.median(pts_b[matches[:, 1]] - pts_a[matches[:, 0]], axis=0)
    else:
        flow_new = np.zeros(3)
    return matches, flow_new


def build_links(frames, cfg):
    """Link consecutive frames into an initial (chain-only) track graph."""
    G = TrackGraph()
    frame_gids = []
    for t, fr in enumerate(frames):
        frame_gids.append([G.add_node(t, fr["um"][i], fr["vox"][i])
                           for i in range(len(fr["um"]))])

    flow = np.zeros(3)
    for t in range(len(frames) - 1):
        a, b = frames[t]["um"], frames[t + 1]["um"]
        prior = flow if cfg["use_motion_prior"] else None
        matches, new_flow = _match_frames(a, b, cfg["max_link_dist_um"], prior)
        for i, j in matches:
            G.add_edge(frame_gids[t][i], frame_gids[t + 1][j])
        if len(matches):
            # Exponential moving average keeps the motion prior stable.
            flow = 0.5 * flow + 0.5 * new_flow if cfg["use_motion_prior"] else flow
    return G, frame_gids


# --------------------------------------------------------------------------- #
# Gap closing (with optional interpolation of skipped frames)
# --------------------------------------------------------------------------- #
def close_gaps(G, cfg):
    from scipy.spatial import cKDTree

    max_gap = cfg["max_gap"]
    if max_gap < 1:
        return G

    ends = [n for n in range(G.n) if not G.children[n]]        # no forward link
    starts_by_t = {}
    for n in range(G.n):
        if G.parent[n] == -1:
            starts_by_t.setdefault(G.t[n], []).append(n)

    # Candidate (distance, end, start) triples across gaps of 2..max_gap frames.
    cand = []
    for e in ends:
        te, pe = G.t[e], G.um[e]
        for gap in range(2, max_gap + 2):    # gap = ts - te (>=2 means >=1 missed)
            ts = te + gap
            starts = starts_by_t.get(ts)
            if not starts:
                continue
            pts = np.stack([G.um[s] for s in starts])
            tree = cKDTree(pts)
            radius = cfg["gap_dist_um"] * gap
            hits = tree.query_ball_point(pe, r=radius)
            for h in hits:
                cand.append((float(np.linalg.norm(pts[h] - pe)), e, starts[h], gap))
    cand.sort()

    used_end, used_start = set(), set()
    for _, e, s, gap in cand:
        if e in used_end or s in used_start or G.children[e]:
            continue
        used_end.add(e)
        used_start.add(s)
        if cfg["interpolate_gaps"] and gap >= 2:
            prev = e
            for step in range(1, gap):
                frac = step / gap
                um = (1 - frac) * G.um[e] + frac * G.um[s]
                vox = (1 - frac) * G.vox[e] + frac * G.vox[s]
                mid = G.add_node(G.t[e] + step, um, vox, interp=True)
                G.add_edge(prev, mid)
                prev = mid
            G.add_edge(prev, s)
        else:
            G.add_edge(e, s)
    return G


# --------------------------------------------------------------------------- #
# Division detection
# --------------------------------------------------------------------------- #
def detect_divisions(G, cfg):
    """Turn a parent into a division when two plausible daughters exist at t+1.

    Handles both real cases after one-to-one linking: (a) the parent is a track
    end with two nearby births, and (b) the parent already continues to one
    daughter and a second daughter is a nearby unmatched birth (a "false
    continuation" that is really a split).
    """
    if not cfg["enable_divisions"]:
        return G

    R = cfg["sister_radius_um"]
    pair_max = cfg["sister_pair_max_um"]
    sym_tol = cfg["division_symmetry_tol_um"]

    starts_by_t = {}
    for n in range(G.n):
        if G.parent[n] == -1:
            starts_by_t.setdefault(G.t[n], []).append(n)

    def quality(a_um, b_um, p_um):
        """Lower is better; None if the pair fails the division guards."""
        pair_d = np.linalg.norm(a_um - b_um)
        mid_d = np.linalg.norm((a_um + b_um) / 2.0 - p_um)
        if pair_d <= pair_max and mid_d <= sym_tol:
            return mid_d + 0.25 * pair_d
        return None

    cand = []  # (quality, parent, daughter1, daughter2)
    for p in range(G.n):
        pum = G.um[p]
        births = [s for s in starts_by_t.get(G.t[p] + 1, [])
                  if G.parent[s] == -1 and np.linalg.norm(G.um[s] - pum) <= R]
        cont = [c for c in G.children[p] if G.t[c] == G.t[p] + 1]
        if cont:  # false-continuation split: existing daughter + one birth
            c = cont[0]
            best = None
            for s in births:
                q = quality(G.um[c], G.um[s], pum)
                if q is not None and (best is None or q < best[0]):
                    best = (q, s)
            if best is not None:
                cand.append((best[0], p, c, best[1]))
        elif len(births) >= 2:  # end with two births
            best = None
            for i in range(len(births)):
                for j in range(i + 1, len(births)):
                    q = quality(G.um[births[i]], G.um[births[j]], pum)
                    if q is not None and (best is None or q < best[0]):
                        best = (q, births[i], births[j])
            if best is not None:
                cand.append((best[0], p, best[1], best[2]))

    cand.sort(key=lambda item: item[0])
    used = set()
    for _, p, d1, d2 in cand:
        if p in used or d1 in used or d2 in used or len(G.children[p]) >= 2:
            continue
        if G.parent[d1] not in (-1, p) or G.parent[d2] not in (-1, p):
            continue
        if G.parent[d1] == -1:
            G.add_edge(p, d1)
        if G.parent[d2] == -1:
            G.add_edge(p, d2)
        used.update((p, d1, d2))
    return G


# --------------------------------------------------------------------------- #
# Track assembly
# --------------------------------------------------------------------------- #
def assemble(G, cfg):
    """Split the graph at divisions into tracks; return a tidy per-node frame."""
    children, parent = G.children, G.parent

    def is_start(n):
        p = parent[n]
        return p == -1 or len(children[p]) >= 2

    # Pass 1: label every node with a track id by walking single-child chains.
    track_of = [-1] * G.n
    chains = []
    tid = 0
    for n in range(G.n):
        if not is_start(n):
            continue
        chain, cur = [], n
        while True:
            chain.append(cur)
            track_of[cur] = tid
            ch = children[cur]
            if len(ch) == 1:
                cur = ch[0]
            else:
                break
        chains.append((tid, chain, parent[n]))
        tid += 1

    # Pass 2: resolve each track's parent track id (valid regardless of order).
    rows = []
    for track_id, chain, parent_node in chains:
        parent_track = track_of[parent_node] if parent_node != -1 else -1
        for node in chain:
            z, y, x = G.vox[node]
            zu, yu, xu = G.um[node]
            rows.append((track_id, G.t[node], z, y, x, zu, yu, xu, parent_track))
    df = pd.DataFrame(rows, columns=["track_id", "t", "z", "y", "x",
                                     "z_um", "y_um", "x_um", "parent_track_id"])
    return df


def filter_short_tracks(df, cfg):
    """Drop isolated tracks shorter than min_track_len, preserving lineages."""
    min_len = cfg["min_track_len"]
    if min_len <= 1 or df.empty:
        return df
    lengths = df.groupby("track_id").size()
    parents = set(df.loc[df.parent_track_id != -1, "parent_track_id"].unique())
    has_parent = set(df.loc[df.parent_track_id != -1, "track_id"].unique())
    removable = {tid for tid, ln in lengths.items()
                 if ln < min_len and tid not in parents and tid not in has_parent}
    if removable:
        df = df[~df.track_id.isin(removable)].copy()
    # Relabel track ids to a contiguous range and remap parent references.
    uniq = {old: new for new, old in enumerate(sorted(df.track_id.unique()))}
    df["track_id"] = df.track_id.map(uniq)
    df["parent_track_id"] = df.parent_track_id.map(lambda p: uniq.get(p, -1))
    return df.sort_values(["track_id", "t"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Submission writing -- match the sample schema discovered at runtime
# --------------------------------------------------------------------------- #
ALIASES = {
    "track_id": ["track_id", "trackid", "track", "id", "label", "cell_id", "cellid"],
    "t": ["t", "time", "frame", "timepoint", "tp", "time_point"],
    "z": ["z", "pos_z", "centroid_z", "z_um", "zc"],
    "y": ["y", "pos_y", "centroid_y", "y_um", "yc"],
    "x": ["x", "pos_x", "centroid_x", "x_um", "xc"],
    "parent_track_id": ["parent_track_id", "parent_id", "parent", "parent_label",
                        "parenttrackid", "parentid"],
}


def infer_columns(sample_cols):
    lower = {c.lower(): c for c in sample_cols}
    mapping = {}
    for canon, opts in ALIASES.items():
        for o in opts:
            if o in lower:
                mapping[canon] = lower[o]
                break
    known = set(mapping.values())
    extra = [c for c in sample_cols if c not in known]  # e.g. a dataset/fov id
    return mapping, extra


def find_sample_submission(input_root):
    hits = glob.glob(os.path.join(input_root, "**", "sample_submission.csv"), recursive=True)
    if not hits:
        return None, None
    hits.sort(key=len)
    return pd.read_csv(hits[0]), hits[0]


def to_submission(tracks, sample_df, cfg, dataset_col_value=None):
    """Map assembled tracks onto the sample_submission column names/order."""
    mapping, extra = infer_columns(list(sample_df.columns))
    out = pd.DataFrame(index=range(len(tracks)))

    if "track_id" in mapping:
        out[mapping["track_id"]] = tracks["track_id"].values
    if "t" in mapping:
        out[mapping["t"]] = tracks["t"].astype(int).values

    cz, cy, cx = ("z_um", "y_um", "x_um") if cfg.get("coord_mode") == "um" else ("z", "y", "x")
    if "z" in mapping:
        out[mapping["z"]] = tracks[cz].values.astype(float)
    if "y" in mapping:
        out[mapping["y"]] = tracks[cy].values.astype(float)
    if "x" in mapping:
        out[mapping["x"]] = tracks[cx].values.astype(float)
    if "parent_track_id" in mapping:
        out[mapping["parent_track_id"]] = tracks["parent_track_id"].astype(int).values

    for col in extra:
        if dataset_col_value is not None:
            out[col] = dataset_col_value
        elif len(sample_df):
            out[col] = sample_df[col].iloc[0]
        else:
            out[col] = 0

    # Preserve the sample's exact column order; fill any we could not map.
    for c in sample_df.columns:
        if c not in out.columns:
            out[c] = 0
    return out[list(sample_df.columns)]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _group_value_for(name, group_values, k):
    """Pick the sample's grouping id for a dataset: prefer a name match (the
    zarr store is usually named after the sequence), else positional, else name.
    """
    for gv in group_values:
        s = str(gv)
        if s == name or s in name or name in s:
            return gv
    if k < len(group_values):
        return group_values[k]
    return name


def track_dataset(ds, cfg):
    frames = detect_all(ds, cfg)
    G, _ = build_links(frames, cfg)
    G = close_gaps(G, cfg)
    G = detect_divisions(G, cfg)
    df = assemble(G, cfg)
    df = filter_short_tracks(df, cfg)
    daughters = int((df.parent_track_id != -1).groupby(df.track_id).any().sum()) if not df.empty else 0
    log(f"  '{ds.name}': {df.track_id.nunique()} tracks, {len(df)} nodes, "
        f"{daughters} daughter tracks (~{daughters // 2} divisions)", cfg)
    return df


def run_pipeline(input_root="/kaggle/input", out_path="submission.csv", cfg=CONFIG):
    t0 = time.time()
    zarr = import_zarr()
    sample_df, sample_path = find_sample_submission(input_root)
    if sample_df is None:
        raise FileNotFoundError(
            f"sample_submission.csv not found under {input_root}. The submission "
            "schema is discovered from it at runtime."
        )
    log(f"sample_submission columns: {list(sample_df.columns)} ({sample_path})", cfg)
    mapping, extra = infer_columns(list(sample_df.columns))
    log(f"column mapping: {mapping}  extra/grouping: {extra}", cfg)

    datasets = discover_datasets(zarr, input_root, cfg)
    if not datasets:
        raise FileNotFoundError(f"No OME-Zarr test volumes found under {input_root}.")

    # Map each dataset to a grouping value if the sample carries one.
    group_values = None
    if extra:
        vals = sample_df[extra[0]].dropna().unique().tolist()
        group_values = vals if vals else None

    parts, base = [], 0
    for k, ds in enumerate(datasets):
        tracks = track_dataset(ds, cfg)
        if not tracks.empty:
            tracks = tracks.copy()
            tracks["track_id"] += base
            tracks.loc[tracks.parent_track_id != -1, "parent_track_id"] += base
            base = int(tracks.track_id.max()) + 1
        gval = _group_value_for(ds.name, group_values, k) if group_values else None
        parts.append(to_submission(tracks, sample_df, cfg, dataset_col_value=gval))

    submission = pd.concat(parts, ignore_index=True) if parts else sample_df.iloc[0:0]
    _validate(submission, sample_df)
    submission.to_csv(out_path, index=False)
    log(f"wrote {out_path}: {len(submission)} rows, {time.time() - t0:.1f}s total", cfg)
    return submission


def _validate(sub, sample_df):
    assert list(sub.columns) == list(sample_df.columns), (
        f"column mismatch: {list(sub.columns)} vs {list(sample_df.columns)}")
    assert len(sub) > 0, "submission is empty"
    assert not sub.isnull().values.any(), "submission contains NaNs"


if __name__ == "__main__":
    run_pipeline("/kaggle/input", "submission.csv", CONFIG)
