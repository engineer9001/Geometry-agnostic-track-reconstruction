import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import hashlib
import os
import glob


def _fingerprint_hit_group(hit_ids, t_diffs, edeps):
    m = hashlib.sha1()
    # concatenate bytes in a deterministic way
    for h, t, e in zip(hit_ids, t_diffs, edeps):
        m.update(str(h).encode('utf-8'))
        m.update(b'|')
        m.update(np.float32(t).tobytes())
        m.update(b'|')
        m.update(np.float32(e).tobytes())
        m.update(b'\n')
    return m.hexdigest()


def aggregate_hits_to_hdf5(hit_df, h5_path, group_root='tracks', detect_duplicates=False):
    """
    Write per-track HDF5 groups. Each track becomes a group under `/{group_root}`
    with attributes `event_index`, `track_index`, `true_mom_x`, `true_mom_y`, `true_mom_z`, 
    and a dataset named `hits` containing a structured array of fields: 
    `hit_id` (utf-8 string), `t_diff` (float32), and `edep` (float32).

    This stores only the actual hits present for each track (no wide padding).
    """

    # Validate that the 3D momentum vector columns exist
    mom_cols = ['true_mom_x', 'true_mom_y', 'true_mom_z']
    has_momentum = all(col in hit_df.columns for col in mom_cols)
    if not has_momentum:
        print(f"WARNING: DataFrame is missing one or more momentum columns ({mom_cols}). Momentum will not be stored in HDF5.")
    
    # Open HDF5 file
    with h5py.File(h5_path, 'w') as f:
        root = f.require_group(group_root)

        grouped = hit_df.groupby(['event_index', 'track_index'])
        fingerprints = {} if detect_duplicates else None
        momentum_stats = {'stored': 0, 'missing': 0, 'nan': 0}
        
        for i, ((event_idx, track_idx), group) in enumerate(tqdm(grouped, desc='Writing HDF5 tracks', unit='track')):
            gname = f"track_{i}"
            g = root.create_group(gname)

            # Store attributes
            try:
                g.attrs['event_index'] = str(event_idx)
            except Exception:
                g.attrs['event_index'] = event_idx
            g.attrs['track_index'] = int(track_idx)
            
            # Store true 3D momentum components with validation
            if has_momentum:
                px = group['true_mom_x'].iloc[0]
                py = group['true_mom_y'].iloc[0]
                pz = group['true_mom_z'].iloc[0]
                
                if np.isnan(px) or np.isnan(py) or np.isnan(pz):
                    momentum_stats['nan'] += 1
                else:
                    g.attrs['true_mom_x'] = float(px)
                    g.attrs['true_mom_y'] = float(py)
                    g.attrs['true_mom_z'] = float(pz)
                    momentum_stats['stored'] += 1
            else:
                momentum_stats['missing'] += 1

            # Build structured array for hits
            n = len(group)
            if n == 0:
                # create empty dataset
                dt = np.dtype([('hit_id', h5py.string_dtype(encoding='utf-8')),
                               ('t_diff', np.float32),
                               ('edep', np.float32)])
                g.create_dataset('hits', shape=(0,), dtype=dt)
                continue

            str_dt = h5py.string_dtype(encoding='utf-8')
            dt = np.dtype([('hit_id', str_dt), ('t_diff', np.float32), ('edep', np.float32)])
            data = np.zeros(n, dtype=dt)

            # Fill fields
            hit_ids = group['hit_id'].astype(str).values
            t_diffs = group['t_diff'].astype(np.float32).values
            edeps = group['edep'].astype(np.float32).values

            data['hit_id'] = hit_ids
            data['t_diff'] = t_diffs
            data['edep'] = edeps

            # Create dataset compressed
            g.create_dataset('hits', data=data, compression='gzip')

            # Duplicate detection (optional)
            if detect_duplicates:
                fp = _fingerprint_hit_group(hit_ids, t_diffs, edeps)
                fingerprints.setdefault(fp, []).append((i, event_idx, track_idx))

        # Report momentum statistics
        print(f"\nMomentum storage summary:")
        print(f"  ✓ Stored: {momentum_stats['stored']} tracks")
        if momentum_stats['nan'] > 0:
            print(f"  ⚠ NaN values (not stored): {momentum_stats['nan']} tracks")
        if momentum_stats['missing'] > 0:
            print(f"  ⚠ Missing momentum columns: {momentum_stats['missing']} tracks")

        # After writing all groups, report duplicates if requested
        if detect_duplicates:
            dupes = {k: v for k, v in fingerprints.items() if len(v) > 1}
            if dupes:
                print('\nDuplicate track fingerprints detected:')
                for fp, entries in dupes.items():
                    print(f'Fingerprint {fp} occurs {len(entries)} times:')
                    for (idx, ev, tr) in entries:
                        print(f'  - group_index={idx}, event={ev}, track={tr}')
                    # Print small diagnostic samples from the original DataFrame
                    print('\nDiagnostic samples for this fingerprint:')
                    for (idx, ev, tr) in entries:
                        # select rows matching this event/track in the original hit_df
                        sel = hit_df[(hit_df['event_index'] == ev) & (hit_df['track_index'] == tr)]
                        print(f'-- event={ev}, track={tr}, rows={len(sel)}')
                        if len(sel) > 0:
                            print(sel[['hit_id', 't_diff', 'edep']].head(5).to_string(index=False))
                        print('')
            else:
                print('\nNo duplicate track fingerprints found.')


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
    # Count unique track indices per event
    per_event_counts = hit_df.groupby('event_index')['track_index'].nunique()
    n_events = per_event_counts.shape[0]
    n_single = int((per_event_counts == 1).sum())
    all_single = (n_single == n_events)

    # If not all single-track events, we can still report stats
    if not all_single:
        return False, False, {'n_events': n_events, 'n_single_track_events': n_single}

    # For events with a single track, compute fingerprint per event
    fps = {}
    for event_idx, group in hit_df.groupby('event_index'):
        # group may contain multiple rows for the single track
        # determine the unique track_index
        track_idxs = group['track_index'].unique()
        if len(track_idxs) != 1:
            # unexpected but treat as not single
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
        
        # Check for duplicates within this event
        dupe_count = sum(1 for v in track_fps.values() if len(v) > 1)
        if dupe_count > 0:
            n_multi_events_with_dupes += 1
            n_dupe_pairs += dupe_count
    
    return {'multi_events': len(multi_events), 'multi_events_with_dupes': n_multi_events_with_dupes, 'dupe_track_pairs': n_dupe_pairs}



if __name__ == '__main__':

    from uproot_data_extractor import uproot_data_extractor
    mldata_dir = '/exp/mu2e/data/users/dgmyers/MLData'
    root_files = sorted(glob.glob(os.path.join(mldata_dir, 'rootFiles', '*.root')))
    
    if not root_files:
        print(f"No .root files found in {mldata_dir}")
    else:
        print(f"Found {len(root_files)} .root file(s) in {mldata_dir}\n")
        
        for root_file in root_files:
            root_basename = os.path.basename(root_file)
            h5_basename = root_basename.replace('.root', '.h5')
            out_h5 = os.path.join(mldata_dir, 'h5Files/', h5_basename)
            
            print(f"Processing: {root_basename}")
            hit_df = uproot_data_extractor(root_file)
            
            if hit_df.empty:
                print(f"  Skipped (empty dataframe)\n")
                continue
            
            # Write HDF5
            aggregate_hits_to_hdf5(hit_df, out_h5, detect_duplicates=False)
            
            # Summary
            print(f"  Wrote {h5_basename}")