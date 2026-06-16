import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import hashlib
import os
import glob


# Hit-level feature columns written into the hits dataset.
# These are the per-hit quantities that vary within a track.
HIT_FEATURE_COLS = [
    'hit_id',       # string — channel identifier (plane_panel_layer_straw)
    't_diff',       # float32 — time difference between the two wire ends [ns]
    'edep',         # float32 — energy deposit [MeV]
    'x_position',   # float32 — POCA x [mm]
    'y_position',   # float32 — POCA y [mm]
    'z_position',   # float32 — POCA z [mm]
    'hit_rho',      # float32 — transverse radius sqrt(x²+y²) [mm]
    'hit_position', # float32 — 3D radius sqrt(x²+y²+z²) [mm]
]

# Track-level scalar attributes stored as HDF5 group attributes.
# These are constant for all hits in a track.
TRACK_SCALAR_ATTRS = [
    # MC truth
    'true_mom_x',    # float32 — true initial momentum x [MeV/c]
    'true_mom_y',    # float32 — true initial momentum y [MeV/c]
    'true_mom_z',    # float32 — true initial momentum z [MeV/c]
    # Calorimeter cluster (NaN if no match)
    'calo_matched',  # bool — True if a calo cluster was matched
    'calo_edep',     # float32 — cluster energy deposit [MeV]
    'calo_edeperr',  # float32 — cluster energy deposit uncertainty [MeV]
    'calo_ctime',    # float32 — cluster time [ns]
    'calo_ctimeerr', # float32 — cluster time uncertainty [ns]
    'calo_did',      # float32 — disk ID (0 or 1; NaN if no match)
    'calo_doca',     # float32 — distance of closest approach [mm]
    'calo_dt',       # float32 — time residual [ns]
    'calo_dphidot',  # float32 — angular velocity dot product
    'calo_ptoca',    # float32 — path length to closest approach [mm]
    'calo_tocavar',  # float32 — time-of-closest-approach variance
    'calo_tresid',   # float32 — time residual
    'calo_tresidmvar',# float32 — time residual minus variance
    'calo_tresidpvar',# float32 — time residual plus variance
    'calo_cdepth',   # float32 — cluster depth [mm]
    'calo_trkdepth', # float32 — track depth at cluster [mm]
    'calo_csize',    # float32 — cluster size (number of crystals)
    'calo_poca_x',   # float32 — POCA x at calo [mm]
    'calo_poca_y',   # float32 — POCA y at calo [mm]
    'calo_poca_z',   # float32 — POCA z at calo [mm]
    'calo_mom_x',    # float32 — track momentum x at calo [MeV/c]
    'calo_mom_y',    # float32 — track momentum y at calo [MeV/c]
    'calo_mom_z',    # float32 — track momentum z at calo [MeV/c]
]


def _fingerprint_hit_group(hit_ids, t_diffs, edeps):
    m = hashlib.sha1()
    for h, t, e in zip(hit_ids, t_diffs, edeps):
        m.update(str(h).encode('utf-8'))
        m.update(b'|')
        m.update(np.float32(t).tobytes())
        m.update(b'|')
        m.update(np.float32(e).tobytes())
        m.update(b'\n')
    return m.hexdigest()


def aggregate_hits_to_hdf5(hit_df, h5_path, group_root='tracks',
                            detect_duplicates=False, calo_hits_dict=None):
    """
    Write per-track HDF5 groups. Each track becomes a group under `/{group_root}`.

    Group attributes (track-level scalars):
        event_index, track_index
        true_mom_x, true_mom_y, true_mom_z   (MC truth)
        calo_matched, calo_edep, calo_ctime, calo_did, ...  (calorimeter cluster)

    Dataset 'hits' (structured array, one row per hit):
        hit_id      : utf-8 string  (plane_panel_layer_straw)
        t_diff      : float32
        edep        : float32
        x_position  : float32
        y_position  : float32
        z_position  : float32
        hit_rho     : float32
        hit_position: float32

    Dataset 'calo_hits' (structured array, one row per crystal hit, optional):
        crystal_id  : int32   — crystal channel ID (0–1338)
        edep        : float32 — energy deposit [MeV]
        edep_err    : float32 — energy deposit uncertainty [MeV]
        time        : float32 — hit time [ns]
        time_err    : float32 — hit time uncertainty [ns]
        n_sipms     : int32   — number of SiPMs that fired
        pos_x       : float32 — crystal centre x [mm]
        pos_y       : float32 — crystal centre y [mm]
        pos_z       : float32 — crystal centre z [mm]
        Written only when calo_hits_dict is provided and the track has a calo match.
        Unmatched tracks get an empty (shape 0) calo_hits dataset.
    """

    # Determine which hit feature columns are actually present
    available_hit_cols = [c for c in HIT_FEATURE_COLS if c in hit_df.columns]
    available_scalar_attrs = [c for c in TRACK_SCALAR_ATTRS if c in hit_df.columns]

    missing_hit = set(HIT_FEATURE_COLS) - set(available_hit_cols)
    missing_scalar = set(TRACK_SCALAR_ATTRS) - set(available_scalar_attrs)
    if missing_hit:
        print(f"WARNING: Missing hit-level columns (will be omitted): {sorted(missing_hit)}")
    if missing_scalar:
        print(f"WARNING: Missing track-level scalar columns (will be omitted): {sorted(missing_scalar)}")

    has_momentum = all(c in hit_df.columns for c in ['true_mom_x', 'true_mom_y', 'true_mom_z'])
    has_calo = 'calo_matched' in hit_df.columns

    with h5py.File(h5_path, 'w') as f:
        root = f.require_group(group_root)

        # Store metadata about what's in this file
        root.attrs['hit_feature_cols']    = available_hit_cols
        root.attrs['track_scalar_attrs']  = available_scalar_attrs
        root.attrs['has_momentum']        = has_momentum
        root.attrs['has_calo']            = has_calo
        root.attrs['has_calo_hits']       = (calo_hits_dict is not None)

        grouped = hit_df.groupby(['event_index', 'track_index'])
        fingerprints = {} if detect_duplicates else None
        stats = {'stored': 0, 'nan_mom': 0, 'calo_matched': 0, 'calo_unmatched': 0}

        # Dtype for per-crystal calo hits dataset
        CALO_HIT_DTYPE = np.dtype([
            ('crystal_id', np.int32),
            ('edep',       np.float32),
            ('edep_err',   np.float32),
            ('time',       np.float32),
            ('time_err',   np.float32),
            ('n_sipms',    np.int32),
            ('pos_x',      np.float32),
            ('pos_y',      np.float32),
            ('pos_z',      np.float32),
        ])

        for i, ((event_idx, track_idx), group) in enumerate(tqdm(grouped, desc='Writing HDF5 tracks', unit='track')):
            gname = f"track_{i}"
            g = root.create_group(gname)

            # --- Track-level attributes ---
            try:
                g.attrs['event_index'] = str(event_idx)
            except Exception:
                g.attrs['event_index'] = event_idx
            g.attrs['track_index'] = int(track_idx)

            # MC momentum
            if has_momentum:
                px = group['true_mom_x'].iloc[0]
                py = group['true_mom_y'].iloc[0]
                pz = group['true_mom_z'].iloc[0]
                if not (np.isnan(px) or np.isnan(py) or np.isnan(pz)):
                    g.attrs['true_mom_x'] = float(px)
                    g.attrs['true_mom_y'] = float(py)
                    g.attrs['true_mom_z'] = float(pz)
                    stats['stored'] += 1
                else:
                    stats['nan_mom'] += 1

            # Calorimeter scalars
            if has_calo:
                c_matched = bool(group['calo_matched'].iloc[0])
                g.attrs['calo_matched'] = c_matched
                if c_matched:
                    stats['calo_matched'] += 1
                else:
                    stats['calo_unmatched'] += 1

                for col in available_scalar_attrs:
                    if col in ('true_mom_x', 'true_mom_y', 'true_mom_z', 'calo_matched'):
                        continue  # already handled above
                    val = group[col].iloc[0]
                    # Store NaN as a special float — h5py handles float NaN fine
                    g.attrs[col] = float(val) if not isinstance(val, bool) else bool(val)

            # --- Hit-level dataset ---
            n = len(group)
            if n == 0:
                # Build empty dtype and create empty dataset
                dt = _build_hit_dtype(available_hit_cols)
                g.create_dataset('hits', shape=(0,), dtype=dt)
                continue

            dt = _build_hit_dtype(available_hit_cols)
            data = np.zeros(n, dtype=dt)

            for col in available_hit_cols:
                if col == 'hit_id':
                    data['hit_id'] = group['hit_id'].astype(str).values
                else:
                    data[col] = group[col].astype(np.float32).values

            g.create_dataset('hits', data=data, compression='gzip')

            # --- Per-crystal calo hits dataset ---
            if calo_hits_dict is not None:
                crystal_arr = calo_hits_dict.get((str(event_idx), int(track_idx)),
                              calo_hits_dict.get((event_idx, track_idx), None))
                if crystal_arr is None:
                    crystal_arr = np.zeros(0, dtype=CALO_HIT_DTYPE)
                g.create_dataset('calo_hits', data=crystal_arr, compression='gzip')

            # Duplicate detection (optional)
            if detect_duplicates:
                hit_ids = group['hit_id'].astype(str).values
                t_diffs = group['t_diff'].astype(np.float32).values
                edeps   = group['edep'].astype(np.float32).values
                fp = _fingerprint_hit_group(hit_ids, t_diffs, edeps)
                fingerprints.setdefault(fp, []).append((i, event_idx, track_idx))

        # Summary
        n_written = stats['stored'] + stats['nan_mom'] + stats['calo_unmatched'] + stats['calo_matched']
        print(f"\nTrack storage summary:")
        print(f"  ✓ Tracks written: {n_written}")
        if has_momentum:
            print(f"  ✓ Momentum stored: {stats['stored']} tracks")
            if stats['nan_mom'] > 0:
                print(f"  ⚠ NaN momentum (not stored): {stats['nan_mom']} tracks")
        if has_calo:
            print(f"  ✓ Calo matched: {stats['calo_matched']} tracks")
            print(f"  ✓ Calo unmatched: {stats['calo_unmatched']} tracks")
        if calo_hits_dict is not None:
            n_with_crystals = sum(1 for v in calo_hits_dict.values() if len(v) > 0)
            print(f"  ✓ Tracks with crystal hits: {n_with_crystals}")

        if detect_duplicates:
            dupes = {k: v for k, v in fingerprints.items() if len(v) > 1}
            if dupes:
                print('\nDuplicate track fingerprints detected:')
                for fp, entries in dupes.items():
                    print(f'Fingerprint {fp} occurs {len(entries)} times:')
                    for (idx, ev, tr) in entries:
                        print(f'  - group_index={idx}, event={ev}, track={tr}')
                    print('\nDiagnostic samples for this fingerprint:')
                    for (idx, ev, tr) in entries:
                        sel = hit_df[(hit_df['event_index'] == ev) & (hit_df['track_index'] == tr)]
                        print(f'-- event={ev}, track={tr}, rows={len(sel)}')
                        if len(sel) > 0:
                            print(sel[['hit_id', 't_diff', 'edep']].head(5).to_string(index=False))
                        print('')
            else:
                print('\nNo duplicate track fingerprints found.')


def _build_hit_dtype(available_hit_cols):
    """Build the numpy structured dtype for the hits dataset."""
    fields = []
    for col in available_hit_cols:
        if col == 'hit_id':
            fields.append(('hit_id', h5py.string_dtype(encoding='utf-8')))
        else:
            fields.append((col, np.float32))
    return np.dtype(fields)


def check_single_track_duplication(hit_df):
    """
    Silent check to determine if each event contains exactly one track, and
    whether those per-event tracks are all identical across events.

    Returns a tuple (all_single, identical_across_events, summary_dict)
    where:
      - all_single: True if every event has exactly one unique track_index
      - identical_across_events: True if all the per-event single-track fingerprints are identical
      - summary_dict: small stats (n_events, n_single_track_events, n_unique_fingerprints)
    """
    per_event_counts = hit_df.groupby('event_index')['track_index'].nunique()
    n_events = per_event_counts.shape[0]
    n_single = int((per_event_counts == 1).sum())
    all_single = (n_single == n_events)

    if not all_single:
        return False, False, {'n_events': n_events, 'n_single_track_events': n_single}

    fps = {}
    for event_idx, group in hit_df.groupby('event_index'):
        track_idxs = group['track_index'].unique()
        if len(track_idxs) != 1:
            return False, False, {'n_events': n_events, 'n_single_track_events': n_single}
        t_idx = track_idxs[0]
        sel = group[group['track_index'] == t_idx]
        hit_ids = sel['hit_id'].astype(str).values
        t_diffs = sel['t_diff'].astype(np.float32).values
        edeps = sel['edep'].astype(np.float32).values
        fp = _fingerprint_hit_group(hit_ids, t_diffs, edeps)
        fps.setdefault(fp, []).append(event_idx)

    n_unique_fps = len(fps)
    identical_across_events = (n_unique_fps == 1)
    return True, identical_across_events, {'n_events': n_events, 'n_single_track_events': n_single, 'n_unique_fingerprints': n_unique_fps}


def check_multi_track_duplication(hit_df):
    """
    For events with >1 track, check if any of those tracks have identical fingerprints
    within the same event. Returns count of such events and total duplicate pairs found.
    """
    per_event_counts = hit_df.groupby('event_index')['track_index'].nunique()
    multi_events = per_event_counts[per_event_counts > 1].index

    n_multi_events_with_dupes = 0
    n_dupe_pairs = 0

    for event_idx in multi_events:
        group = hit_df[hit_df['event_index'] == event_idx]
        track_fps = {}
        for track_idx, tgroup in group.groupby('track_index'):
            hit_ids = tgroup['hit_id'].astype(str).values
            t_diffs = tgroup['t_diff'].astype(np.float32).values
            edeps = tgroup['edep'].astype(np.float32).values
            fp = _fingerprint_hit_group(hit_ids, t_diffs, edeps)
            track_fps.setdefault(fp, []).append(track_idx)

        dupe_count = sum(1 for v in track_fps.values() if len(v) > 1)
        if dupe_count > 0:
            n_multi_events_with_dupes += 1
            n_dupe_pairs += dupe_count

    return {'multi_events': len(multi_events), 'multi_events_with_dupes': n_multi_events_with_dupes, 'dupe_track_pairs': n_dupe_pairs}


if __name__ == '__main__':

    from uproot_data_extractor import uproot_data_extractor
    mldata_dir = '/exp/mu2e/app/users/dgmyers/MLWork_EAF/DOA/MLData'
    h5_out_dir = os.path.join(mldata_dir, 'h5Files')
    root_files = sorted(glob.glob(os.path.join(mldata_dir, 'rootFiles', '*.root')))

    if not root_files:
        print(f"No .root files found in {mldata_dir}/rootFiles/")
    else:
        # Ensure output directory exists
        os.makedirs(h5_out_dir, exist_ok=True)
        print(f"Found {len(root_files)} .root file(s)")
        print(f"Output directory: {h5_out_dir}\n")

        for root_file in root_files:
            root_basename = os.path.basename(root_file)
            h5_basename = root_basename.replace('.root', '.h5')
            out_h5 = os.path.join(h5_out_dir, h5_basename)

            # Skip if already done
            if os.path.exists(out_h5):
                print(f"  Already exists, skipping: {h5_basename}")
                continue

            print(f"Processing: {root_basename}")
            hit_df, calo_hits_dict = uproot_data_extractor(root_file)

            if hit_df.empty:
                print(f"  Skipped (empty dataframe)\n")
                continue

            aggregate_hits_to_hdf5(hit_df, out_h5, detect_duplicates=False,
                                   calo_hits_dict=calo_hits_dict)
            print(f"  Wrote: {out_h5}\n")

        print("\nAll files processed.")
