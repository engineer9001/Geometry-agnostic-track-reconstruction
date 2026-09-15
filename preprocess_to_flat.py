import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import h5py
from tqdm import tqdm

from track_aggregation import TRACK_SCALAR_ATTRS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CALO_SCALAR_COLS = [
    c for c in TRACK_SCALAR_ATTRS
    if c not in ('true_mom_x', 'true_mom_y', 'true_mom_z', 'calo_matched')
]

MAX_CRYSTALS = 20

# Default quality cuts applied at flattening time.  Tracks that fail these
# cuts are dropped entirely from the flat file (they cost nothing at train
# time then).
#
# The mu2e tracker layout is 36 planes × 6 panels × 2 layers × 96 straws.
# Stations are pairs of planes (station = plane // 2), so there are 18
# stations total.  Requiring ≥2 stations means the track has hits in at
# least 2 different plane-pairs, which excludes noise/short tracks that
# only ping a single station.
DEFAULT_MIN_HITS     = 5
DEFAULT_MIN_STATIONS = 2

# Channel-index → (plane, station) helpers.  Match build_standard_channel_index:
#   idx = plane * (6*2*96) + panel * (2*96) + layer * 96 + straw
_CH_PER_PLANE = 6 * 2 * 96   # 1152 channels per plane


def _plane_from_ch(ch_idx: np.ndarray) -> np.ndarray:
    return ch_idx // _CH_PER_PLANE


def _station_from_ch(ch_idx: np.ndarray) -> np.ndarray:
    return _plane_from_ch(ch_idx) // 2


def build_standard_channel_index() -> Dict[str, int]:
    """Build the standard 41,472-channel index (36 planes × 6 panels × 2 layers × 96 straws)."""
    return {
        f"{p}_{pa}_{l}_{s}": idx
        for idx, (p, pa, l, s) in enumerate(
            [(p, pa, l, s) for p in range(36) for pa in range(6) for l in range(2) for s in range(96)]
        )
    }


def _write_flat_h5(
    out_path: str,
    features: np.ndarray,
    ch_indices: np.ndarray,
    offsets: np.ndarray,
    mom_xyz: Optional[np.ndarray],
    calo_scalars: np.ndarray,
    calo_matched_arr: np.ndarray,
    calo_hits_padded: np.ndarray,
    calo_n_crystals: np.ndarray,
    track_names: List[str],
    feature_names: Tuple[str, ...],
    calo_scalar_cols: List[str],
    max_crystals: int,
) -> None:
    """Write the flat CSR arrays to a new HDF5 file."""
    with h5py.File(out_path, "w") as out:
        out.create_dataset("features",         data=features,          compression="lzf", chunks=True)
        out.create_dataset("ch_indices",       data=ch_indices,        compression="lzf", chunks=True)
        out.create_dataset("offsets",          data=offsets,           compression="lzf", chunks=True)
        out.create_dataset("calo_scalars",     data=calo_scalars,      compression="lzf", chunks=True)
        out.create_dataset("calo_matched",     data=calo_matched_arr,  compression="lzf", chunks=True)
        out.create_dataset("calo_hits_padded", data=calo_hits_padded,  compression="lzf", chunks=True)
        out.create_dataset("calo_n_crystals",  data=calo_n_crystals,   compression="lzf", chunks=True)

        if mom_xyz is not None:
            out.create_dataset("mom_xyz", data=mom_xyz, compression="lzf", chunks=True)

        dt = h5py.string_dtype()
        names_ds = out.create_dataset("track_names", (len(track_names),), dtype=dt)
        for i, name in enumerate(track_names):
            names_ds[i] = name

        out.attrs["n_tracks"]           = len(track_names)
        out.attrs["n_features"]         = features.shape[1] if len(features) > 0 else len(feature_names)
        out.attrs["n_calo_scalars"]     = len(calo_scalar_cols)
        out.attrs["max_crystals"]       = max_crystals
        out.attrs["n_calo_hit_feats"]   = calo_hits_padded.shape[2] if len(calo_hits_padded) > 0 else 3
        out.attrs["feature_names"]      = list(feature_names)
        out.attrs["calo_scalar_cols"]   = calo_scalar_cols
        # Per-crystal features:  [crystal_id, edep, time]  -- time is
        # already anchored to the earliest-hit clock (see track_aggregation.py
        # L204).  This is the ONLY calo information the network sees when
        # --use-calo is enabled; the 22-vector calo_scalars is kept in the
        # file for offline analysis but is no longer fed to the model
        # (it contained reconstruction-output features such as calo_mom_*
        # which effectively leak the target momentum).
        out.attrs["calo_hit_feat_names"]= ["crystal_id", "edep", "time"]
        out.attrs["has_momentum"]       = mom_xyz is not None
        out.attrs["flat_format"]        = True


def _process_single_file(
    args_tuple: Tuple,
) -> Tuple[str, int, int, int, int, int]:
    """
    Worker function: reads one HDF5 file, flattens it, writes it directly, 
    and returns ONLY small summary metrics back to the parent process.

    Quality cut: tracks are dropped from the output entirely if, *after*
    filtering to hits with a valid channel index, they have fewer than
    ``min_hits`` remaining hits or their remaining hits span fewer than
    ``min_stations`` distinct stations (station = plane // 2).  A track
    that has any nan/inf in its momentum target is also dropped (only when
    has_momentum=True), since it cannot supply a regression label.
    """
    (src_path, out_path, channel_index, feature_names, has_momentum,
     calo_scalar_cols, max_crystals, min_hits, min_stations) = args_tuple

    all_features:       List[np.ndarray] = []
    all_indices:        List[np.ndarray] = []
    all_offsets:        List[int]        = [0]
    all_mom:            List[np.ndarray] = []
    all_calo_scalars:   List[np.ndarray] = []
    all_calo_matched:   List[bool]       = []
    all_calo_padded:    List[np.ndarray] = []
    all_calo_n_crys:    List[int]        = []
    track_names:        List[str]        = []
    n_skipped_hits         = 0
    n_rejected_low_hits    = 0
    n_rejected_low_stations = 0
    n_rejected_bad_mom      = 0

    # Per-crystal feature layout: [crystal_id, edep, time]
    n_calo = len(calo_scalar_cols)
    n_calo_hit_feats = 3

    with h5py.File(src_path, "r") as f:
        if "tracks" not in f:
            return (src_path, 0, 0, 0, 0, 0)

        file_has_calo      = f["tracks"].attrs.get("has_calo", False)
        file_has_calo_hits = f["tracks"].attrs.get("has_calo_hits", False)

        for track_name in f["tracks"].keys():
            track_group = f["tracks"][track_name]
            hits = track_group["hits"]

            raw_ids = hits["hit_id"][:]
            hit_ids = [
                hid.decode("utf-8") if isinstance(hid, bytes) else str(hid)
                for hid in raw_ids
            ]
            ch_idx = np.array(
                [channel_index.get(hid, -1) for hid in hit_ids], dtype=np.int32
            )
            valid = ch_idx >= 0
            n_invalid = int((~valid).sum())
            if n_invalid:
                n_skipped_hits += n_invalid
            ch_idx = ch_idx[valid]

            # ---- Quality cuts (applied to the post-channel-filter hits) ----
            if len(ch_idx) < min_hits:
                n_rejected_low_hits += 1
                continue
            n_stations = int(np.unique(_station_from_ch(ch_idx)).size)
            if n_stations < min_stations:
                n_rejected_low_stations += 1
                continue
            # ----------------------------------------------------------------

            # Momentum cut (only when momentum is required by downstream training).
            if has_momentum:
                px = float(track_group.attrs.get("true_mom_x", np.nan))
                py = float(track_group.attrs.get("true_mom_y", np.nan))
                pz = float(track_group.attrs.get("true_mom_z", np.nan))
                if not (np.isfinite(px) and np.isfinite(py) and np.isfinite(pz)):
                    n_rejected_bad_mom += 1
                    continue

            feat_arrays = []
            for name in feature_names:
                if name in hits.dtype.names:
                    feat_arrays.append(np.asarray(hits[name], dtype=np.float32)[valid])
                else:
                    feat_arrays.append(np.zeros(int(valid.sum()), dtype=np.float32))

            feats = (
                np.column_stack(feat_arrays)
                if len(feat_arrays) > 1
                else feat_arrays[0][:, None]
            )

            all_features.append(feats)
            all_indices.append(ch_idx)
            all_offsets.append(all_offsets[-1] + len(ch_idx))
            track_names.append(track_name)

            if has_momentum:
                all_mom.append(np.array([px, py, pz], dtype=np.float32))

            c_matched = bool(track_group.attrs.get("calo_matched", False))
            all_calo_matched.append(c_matched)

            if file_has_calo:
                row = np.array(
                    [float(track_group.attrs.get(col, np.nan)) for col in calo_scalar_cols],
                    dtype=np.float32,
                )
            else:
                row = np.full(n_calo, np.nan, dtype=np.float32)
            all_calo_scalars.append(row)

            padded = np.zeros((max_crystals, n_calo_hit_feats), dtype=np.float32)
            n_crys = 0

            if c_matched and file_has_calo_hits and "calo_hits" in track_group:
                ch_data = track_group["calo_hits"]
                if len(ch_data) > 0:
                    edep_vals = np.asarray(ch_data["edep"], dtype=np.float32)
                    order = np.argsort(edep_vals)[::-1]
                    n_crys = min(len(ch_data), max_crystals)
                    keep = order[:n_crys]
                    padded[:n_crys, 0] = np.asarray(ch_data["crystal_id"], dtype=np.float32)[keep]
                    padded[:n_crys, 1] = edep_vals[keep]
                    # Per-crystal time (already track_t0-anchored upstream in
                    # track_aggregation.py L204).  Any NaN times are left as
                    # 0.0 by the pre-allocated `padded` buffer; the model's
                    # crystal mask will fully hide unpopulated rows.
                    time_vals = np.asarray(ch_data["time"], dtype=np.float32)
                    padded[:n_crys, 2] = time_vals[keep]

            all_calo_padded.append(padded)
            all_calo_n_crys.append(n_crys)

    # Assemble output arrays inside the worker
    n_feat = len(feature_names)
    features = np.concatenate(all_features, axis=0) if all_features else np.empty((0, n_feat), dtype=np.float32)
    ch_indices = np.concatenate(all_indices, axis=0) if all_indices else np.empty(0, dtype=np.int32)
    offsets = np.array(all_offsets, dtype=np.int64)
    mom_xyz = np.stack(all_mom, axis=0) if (has_momentum and all_mom) else (np.empty((0, 3), dtype=np.float32) if has_momentum else None)
    calo_scalars = np.stack(all_calo_scalars, axis=0) if all_calo_scalars else np.empty((0, n_calo), dtype=np.float32)
    calo_matched_arr = np.array(all_calo_matched, dtype=bool)
    calo_hits_padded = np.stack(all_calo_padded, axis=0) if all_calo_padded else np.empty((0, max_crystals, n_calo_hit_feats), dtype=np.float32)
    calo_n_crystals = np.array(all_calo_n_crys, dtype=np.int32)

    # WRITE DIRECTLY FROM THE WORKER PROCESS
    _write_flat_h5(
        out_path, features, ch_indices, offsets, mom_xyz,
        calo_scalars, calo_matched_arr, calo_hits_padded, calo_n_crystals,
        track_names, feature_names, calo_scalar_cols, max_crystals,
    )

    n_tracks = len(offsets) - 1
    n_hits = len(ch_indices)
    n_matched = int(calo_matched_arr.sum())
    n_rejected_total = n_rejected_low_hits + n_rejected_low_stations + n_rejected_bad_mom

    # Return tiny scalar metrics instead of giant memory blocks
    return (src_path, n_tracks, n_hits, n_skipped_hits, n_matched, n_rejected_total)


def convert_directory(
    input_dir: str,
    output_dir: str,
    feature_names: Tuple[str, ...] = ("t0", "t1", "tot0", "tot1"),
    has_momentum: bool = True,
    workers: int = 4,
    max_crystals: int = MAX_CRYSTALS,
    min_hits: int = DEFAULT_MIN_HITS,
    min_stations: int = DEFAULT_MIN_STATIONS,
) -> None:
    """Convert all .h5 files in input_dir to flat CSR format in output_dir.

    Tracks failing the quality cuts (``min_hits`` hits with a valid channel
    index AND ``min_stations`` distinct stations, plus finite momentum when
    has_momentum=True) are dropped and never written to the flat file.
    """
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.h5")))
    if not input_files:
        raise FileNotFoundError(f"No .h5 files found in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    channel_index = build_standard_channel_index()

    logger.info(f"Converting {len(input_files)} files: {input_dir} → {output_dir}")
    logger.info(f"Workers: {workers} | Features: {feature_names} | Momentum: {has_momentum}")
    logger.info(f"Quality cuts: min_hits={min_hits}, min_stations={min_stations}"
                + (", finite momentum required" if has_momentum else ""))

    # Build work items including the target out_path
    work_items = []
    for src in input_files:
        out_name = os.path.basename(src)
        out_path = os.path.join(output_dir, out_name)
        work_items.append((src, out_path, channel_index, feature_names, has_momentum,
                           CALO_SCALAR_COLS, max_crystals, min_hits, min_stations))

    total_tracks = 0
    total_hits = 0
    total_skipped = 0
    total_matched = 0
    total_rejected = 0

    # Execute and monitor
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_single_file, item): item[0]
            for item in work_items
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
            src_path = futures[future]
            try:
                # Unpack the small, lightweight tuple
                _, n_tracks, n_hits, n_skipped, n_matched, n_rejected = future.result()

                total_tracks += n_tracks
                total_hits += n_hits
                total_skipped += n_skipped
                total_matched += n_matched
                total_rejected += n_rejected
            except Exception as e:
                logger.error(f"Failed to convert {src_path}: {e}", exc_info=True)

    total_seen = total_tracks + total_rejected
    logger.info(
        f"Done. {total_tracks:,} tracks kept | {total_rejected:,} rejected by quality cuts "
        f"({100*total_rejected/max(1,total_seen):.2f}% rejection rate) | "
        f"{total_hits:,} tracker hits | {total_skipped:,} hits skipped (unknown channel)"
    )
    if total_tracks:
        logger.info(f"Average tracker hits/track: {total_hits/total_tracks:.1f}")
        logger.info(f"Calo-matched tracks: {total_matched:,} ({100*total_matched/total_tracks:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-process HDF5 track files into flat CSR format for fast DataLoader access.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir",  help="Directory containing source .h5 files")
    parser.add_argument("output_dir", help="Directory to write flat .h5 files")
    parser.add_argument(
        "--feature-names", nargs="+", default=["t0", "t1", "tot0", "tot1"],
        help="Tracker hit-level feature columns to extract",
    )
    parser.add_argument(
        "--no-momentum", action="store_true",
        help="Skip momentum target extraction",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--max-crystals", type=int, default=MAX_CRYSTALS,
        help=f"Fixed size of the per-track calo crystal block (default: {MAX_CRYSTALS}).",
    )
    parser.add_argument(
        "--min-hits", type=int, default=DEFAULT_MIN_HITS,
        help=f"Drop tracks with fewer than this many valid tracker hits after channel filtering (default: {DEFAULT_MIN_HITS}).",
    )
    parser.add_argument(
        "--min-stations", type=int, default=DEFAULT_MIN_STATIONS,
        help=f"Drop tracks whose valid hits span fewer than this many stations (station = plane // 2). Default: {DEFAULT_MIN_STATIONS}.",
    )
    args = parser.parse_args()

    convert_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        feature_names=tuple(args.feature_names),
        has_momentum=not args.no_momentum,
        workers=args.workers,
        max_crystals=args.max_crystals,
        min_hits=args.min_hits,
        min_stations=args.min_stations,
    )


if __name__ == "__main__":
    main()