import uproot
import awkward as ak
import numpy as np
import pandas as pd
import os
import hashlib

# Sentinel value used in TrkAna when a track has no calorimeter match
_CALO_SENTINEL = -1000.0


def _calo_scalar(arr_2d, event_idx, tidx):
    """Extract scalar calo value for track tidx in event event_idx; sentinel → NaN."""
    val = float(arr_2d[event_idx][tidx])
    return np.nan if val <= _CALO_SENTINEL + 1.0 else val


def _calo_int(arr_2d, event_idx, tidx):
    """Extract integer calo value; -1 → NaN."""
    val = int(arr_2d[event_idx][tidx])
    return np.nan if val < 0 else float(val)


def _calo_scalar_gated(arr_2d, event_idx, tidx, matched):
    """Extract scalar; return NaN immediately if track is unmatched.
    Used for fields whose sentinel is 0 rather than -1000 (e.g. poca, mom)."""
    if not matched:
        return np.nan
    return float(arr_2d[event_idx][tidx])


def _track_fingerprint(hit_ids, t_diffs, edeps):
    m = hashlib.sha1()
    for hid, td, ed in zip(hit_ids, t_diffs, edeps):
        m.update(str(hid).encode('utf-8'))
        m.update(b'|')
        m.update(np.float32(td).tobytes())
        m.update(b'|')
        m.update(np.float32(ed).tobytes())
        m.update(b'\n')
    return m.hexdigest()


def uproot_data_extractor(filepaths_string):
    """
    Extracts Mu2e tracker hit data AND calorimeter crystal hit data directly
    from a long string of file paths.

    Returns a tuple: (hit_df, calo_hits_dict)

    hit_df : pd.DataFrame
        One row per tracker straw hit. Columns include hit_id, x/y/z_position,
        hit_rho, hit_position, edep, t_diff, true_mom_x/y/z, calo_matched,
        and all calo_* cluster-level scalars (same value repeated for every
        hit row belonging to that track).

    calo_hits_dict : dict
        Keys are (event_index_str, track_index_int).
        Values are numpy structured arrays with dtype:
            crystal_id  : int32   — crystal channel ID (0–1338)
            edep        : float32 — energy deposit [MeV]
            edep_err    : float32 — energy deposit uncertainty [MeV]
            time        : float32 — hit time [ns]
            time_err    : float32 — hit time uncertainty [ns]
            n_sipms     : int32   — number of SiPMs that fired
            pos_x       : float32 — crystal centre x [mm]
            pos_y       : float32 — crystal centre y [mm]
            pos_z       : float32 — crystal centre z [mm]
        For unmatched tracks the value is an empty array (shape (0,)).

    Calorimeter cluster scalars (track-level, stored in hit_df and as H5 attrs):
        calo_edep, calo_edeperr, calo_ctime, calo_ctimeerr,
        calo_did, calo_doca, calo_dt, calo_dphidot, calo_ptoca,
        calo_tocavar, calo_tresid, calo_tresidmvar, calo_tresidpvar,
        calo_cdepth, calo_trkdepth, calo_csize,
        calo_poca_x/y/z, calo_mom_x/y/z, calo_matched
    """
    feature_variables = []
    calo_hits_dict = {}
    file_counter = 0
    duplicate_tracks_skipped = 0

    file_list = filepaths_string.strip().split()
    if not file_list:
        print("\n\033[91mError: Provided file paths string is empty!\033[0m")
        return pd.DataFrame(), {}

    # Branches to load from the track-indexed part of the tree
    branches_to_load = [
        "trkhits",
        "trkmcsim",
        "trkcalohit",
    ]

    # Branches for per-crystal calorimeter hits (event-level, not track-level)
    calohits_branches = [
        'calohits/calohits.crystalId_',
        'calohits/calohits.eDep_',
        'calohits/calohits.eDepErr_',
        'calohits/calohits.time_',
        'calohits/calohits.timeErr_',
        'calohits/calohits.nSiPMs_',
        'calohits/calohits.crystalPos_.fCoordinates.fX',
        'calohits/calohits.crystalPos_.fCoordinates.fY',
        'calohits/calohits.crystalPos_.fCoordinates.fZ',
    ]

    # Branches for cluster→crystal index mapping
    caloclusters_branches = [
        'caloclusters/caloclusters.energyDep_',
        'caloclusters/caloclusters.time_',
        'caloclusters/caloclusters.hits_',
    ]

    # Structured dtype for per-crystal calo hits stored in H5
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

    for filename in file_list:
        filename = filename.strip()
        if not filename or filename.startswith('#'):
            continue
        file_counter += 1
        if filename.startswith('\ufeff'):
            filename = filename[1:]

        print('\nNow processing file: ', f'\033[93m{filename}\033[0m')

        try:
            with uproot.open(filename) as file:
                tree = file['EventNtuple/ntuple']

                # Load track-indexed branches
                trkana = tree.arrays(expressions=branches_to_load)

                # Load event-level calohits branches
                raw_ch = tree.arrays(expressions=calohits_branches)
                raw_cc = tree.arrays(expressions=caloclusters_branches)

                if len(trkana) == 0:
                    print(f"Warning: File {filename} has an empty tree.")
                    continue

        except Exception as e:
            print(f"\033[91mFailed to read {filename}: {e}\033[0m")
            continue

        # --- Extract Tracker Hit-Level Quantities ---
        hits = trkana["trkhits"]
        plane = hits['plane']
        panel = hits['panel']
        layer = hits['layer']
        straw = hits['straw']

        xpos_targets = hits['poca']['fCoordinates']['fX']
        ypos_targets = hits['poca']['fCoordinates']['fY']
        zpos_targets = hits['poca']['fCoordinates']['fZ']

        rho_hit_targets = np.sqrt(xpos_targets ** 2 + ypos_targets ** 2)
        r_hit_targets   = np.sqrt(xpos_targets ** 2 + ypos_targets ** 2 + zpos_targets ** 2)

        endtime = hits['etime']
        edep    = hits['edep']

        # --- Extract Track-Level MC True INITIAL Momentum Components ---
        mcsim = trkana["trkmcsim"]
        mom_x = mcsim['mom']['fCoordinates']['fX']
        mom_y = mcsim['mom']['fCoordinates']['fY']
        mom_z = mcsim['mom']['fCoordinates']['fZ']

        # --- Extract Track-Level Calorimeter Cluster (one per track) ---
        calo = trkana["trkcalohit"]
        calo_edep      = calo["trkcalohit.edep"]
        calo_edeperr   = calo["trkcalohit.edeperr"]
        calo_ctime     = calo["trkcalohit.ctime"]
        calo_ctimeerr  = calo["trkcalohit.ctimeerr"]
        calo_did       = calo["trkcalohit.did"]
        calo_doca      = calo["trkcalohit.doca"]
        calo_dt        = calo["trkcalohit.dt"]
        calo_dphidot   = calo["trkcalohit.dphidot"]
        calo_ptoca     = calo["trkcalohit.ptoca"]
        calo_tocavar   = calo["trkcalohit.tocavar"]
        calo_tresid    = calo["trkcalohit.tresid"]
        calo_tresidmvar= calo["trkcalohit.tresidmvar"]
        calo_tresidpvar= calo["trkcalohit.tresidpvar"]
        calo_cdepth    = calo["trkcalohit.cdepth"]
        calo_trkdepth  = calo["trkcalohit.trkdepth"]
        calo_csize     = calo["trkcalohit.csize"]
        calo_poca_x    = calo["trkcalohit.poca.fCoordinates.fX"]
        calo_poca_y    = calo["trkcalohit.poca.fCoordinates.fY"]
        calo_poca_z    = calo["trkcalohit.poca.fCoordinates.fZ"]
        calo_mom_x     = calo["trkcalohit.mom.fCoordinates.fX"]
        calo_mom_y     = calo["trkcalohit.mom.fCoordinates.fY"]
        calo_mom_z     = calo["trkcalohit.mom.fCoordinates.fZ"]

        # --- Per-crystal calohit arrays (event-level) ---
        ch_crystalId = raw_ch['calohits/calohits.crystalId_']
        ch_edep      = raw_ch['calohits/calohits.eDep_']
        ch_edeperr   = raw_ch['calohits/calohits.eDepErr_']
        ch_time      = raw_ch['calohits/calohits.time_']
        ch_timeerr   = raw_ch['calohits/calohits.timeErr_']
        ch_nsipms    = raw_ch['calohits/calohits.nSiPMs_']
        ch_pos_x     = raw_ch['calohits/calohits.crystalPos_.fCoordinates.fX']
        ch_pos_y     = raw_ch['calohits/calohits.crystalPos_.fCoordinates.fY']
        ch_pos_z     = raw_ch['calohits/calohits.crystalPos_.fCoordinates.fZ']

        # --- Cluster arrays (event-level) ---
        cc_edep  = raw_cc['caloclusters/caloclusters.energyDep_']
        cc_time  = raw_cc['caloclusters/caloclusters.time_']
        cc_hits  = raw_cc['caloclusters/caloclusters.hits_']  # list of calohit indices per cluster

        # --- Build the Flat DataFrames ---
        for count in range(len(xpos_targets)):
            num_tracks = len(plane[count])
            seen_track_fps = set()

            # Pre-convert event-level calohit arrays to numpy for fast indexing
            n_calohits_ev = len(ch_crystalId[count])
            if n_calohits_ev > 0:
                ev_ch_crystalId = ak.to_numpy(ch_crystalId[count]).astype(np.int32)
                ev_ch_edep      = ak.to_numpy(ch_edep[count]).astype(np.float32)
                ev_ch_edeperr   = ak.to_numpy(ch_edeperr[count]).astype(np.float32)
                ev_ch_time      = ak.to_numpy(ch_time[count]).astype(np.float32)
                ev_ch_timeerr   = ak.to_numpy(ch_timeerr[count]).astype(np.float32)
                ev_ch_nsipms    = ak.to_numpy(ch_nsipms[count]).astype(np.int32)
                ev_ch_pos_x     = ak.to_numpy(ch_pos_x[count]).astype(np.float32)
                ev_ch_pos_y     = ak.to_numpy(ch_pos_y[count]).astype(np.float32)
                ev_ch_pos_z     = ak.to_numpy(ch_pos_z[count]).astype(np.float32)
            else:
                ev_ch_crystalId = np.array([], dtype=np.int32)

            # Pre-convert cluster arrays for this event
            n_clusters_ev = len(cc_edep[count])
            if n_clusters_ev > 0:
                ev_cc_edep = ak.to_numpy(cc_edep[count]).astype(np.float64)
                ev_cc_time = ak.to_numpy(cc_time[count]).astype(np.float64)
            else:
                ev_cc_edep = np.array([], dtype=np.float64)
                ev_cc_time = np.array([], dtype=np.float64)

            for track_idx in range(num_tracks):
                p_arr = ak.to_numpy(plane[count][track_idx])
                if len(p_arr) == 0:
                    continue

                pa_arr  = ak.to_numpy(panel[count][track_idx])
                l_arr   = ak.to_numpy(layer[count][track_idx])
                s_arr   = ak.to_numpy(straw[count][track_idx])

                x_arr   = ak.to_numpy(xpos_targets[count][track_idx])
                y_arr   = ak.to_numpy(ypos_targets[count][track_idx])
                z_arr   = ak.to_numpy(zpos_targets[count][track_idx])
                rho_arr = ak.to_numpy(rho_hit_targets[count][track_idx])
                r_arr   = ak.to_numpy(r_hit_targets[count][track_idx])
                e_arr   = ak.to_numpy(edep[count][track_idx])

                time_entry = ak.to_numpy(endtime[count][track_idx])
                t0_arr     = time_entry[:, 0]
                t1_arr     = time_entry[:, 1]
                t_diff_arr = t1_arr - t0_arr

                hit_id_strings = (
                    p_arr.astype(str) + "_" +
                    pa_arr.astype(str) + "_" +
                    l_arr.astype(str) + "_" +
                    s_arr.astype(str)
                )

                fp = _track_fingerprint(hit_id_strings, t_diff_arr, e_arr)
                if fp in seen_track_fps:
                    duplicate_tracks_skipped += 1
                    continue
                seen_track_fps.add(fp)
                dedup_track_idx = len(seen_track_fps) - 1

                # --- MC true momentum (primary matched particle, index 0) ---
                track_mc_px = mom_x[count][track_idx]
                track_mc_py = mom_y[count][track_idx]
                track_mc_pz = mom_z[count][track_idx]

                if len(track_mc_px) > 0:
                    t_px = float(track_mc_px[0])
                    t_py = float(track_mc_py[0])
                    t_pz = float(track_mc_pz[0])
                else:
                    t_px = np.nan
                    t_py = np.nan
                    t_pz = np.nan

                # --- Calorimeter cluster scalars for this track ---
                c_edep       = _calo_scalar(calo_edep,       count, track_idx)
                c_matched    = not np.isnan(c_edep)

                c_edeperr    = _calo_scalar(calo_edeperr,    count, track_idx)
                c_ctime      = _calo_scalar(calo_ctime,      count, track_idx)
                c_ctimeerr   = _calo_scalar(calo_ctimeerr,   count, track_idx)
                c_did        = _calo_int(calo_did,           count, track_idx)
                c_doca       = _calo_scalar(calo_doca,       count, track_idx)
                c_dt         = _calo_scalar(calo_dt,         count, track_idx)
                c_dphidot    = _calo_scalar(calo_dphidot,    count, track_idx)
                c_ptoca      = _calo_scalar(calo_ptoca,      count, track_idx)
                c_tocavar    = _calo_scalar(calo_tocavar,    count, track_idx)
                c_tresid     = _calo_scalar(calo_tresid,     count, track_idx)
                c_tresidmvar = _calo_scalar(calo_tresidmvar, count, track_idx)
                c_tresidpvar = _calo_scalar(calo_tresidpvar, count, track_idx)
                c_cdepth     = _calo_scalar(calo_cdepth,     count, track_idx)
                c_trkdepth   = _calo_scalar(calo_trkdepth,  count, track_idx)
                c_csize      = _calo_scalar(calo_csize,      count, track_idx)
                c_poca_x     = _calo_scalar_gated(calo_poca_x, count, track_idx, c_matched)
                c_poca_y     = _calo_scalar_gated(calo_poca_y, count, track_idx, c_matched)
                c_poca_z     = _calo_scalar_gated(calo_poca_z, count, track_idx, c_matched)
                c_mom_x      = _calo_scalar_gated(calo_mom_x,  count, track_idx, c_matched)
                c_mom_y      = _calo_scalar_gated(calo_mom_y,  count, track_idx, c_matched)
                c_mom_z      = _calo_scalar_gated(calo_mom_z,  count, track_idx, c_matched)

                # --- Resolve per-crystal hits for this track ---
                # Match the track's cluster by energy+time to find the cluster index,
                # then use caloclusters.hits_ to get the calohit indices.
                crystal_hits = np.zeros(0, dtype=CALO_HIT_DTYPE)

                if c_matched and n_clusters_ev > 0 and n_calohits_ev > 0:
                    # Find the matching cluster (exact match on edep+ctime)
                    de = np.abs(ev_cc_edep - c_edep)
                    dt = np.abs(ev_cc_time - c_ctime)
                    match_mask = (de < 0.01) & (dt < 0.1)
                    match_indices = np.where(match_mask)[0]

                    if len(match_indices) > 0:
                        ci = int(match_indices[0])  # cluster index
                        hit_indices = ak.to_numpy(cc_hits[count][ci]).astype(int)

                        # Filter to valid calohit indices
                        valid = hit_indices[hit_indices < n_calohits_ev]
                        n_crys = len(valid)

                        if n_crys > 0:
                            crystal_hits = np.zeros(n_crys, dtype=CALO_HIT_DTYPE)
                            crystal_hits['crystal_id'] = ev_ch_crystalId[valid]
                            crystal_hits['edep']       = ev_ch_edep[valid]
                            crystal_hits['edep_err']   = ev_ch_edeperr[valid]
                            crystal_hits['time']       = ev_ch_time[valid]
                            crystal_hits['time_err']   = ev_ch_timeerr[valid]
                            crystal_hits['n_sipms']    = ev_ch_nsipms[valid]
                            crystal_hits['pos_x']      = ev_ch_pos_x[valid]
                            crystal_hits['pos_y']      = ev_ch_pos_y[valid]
                            crystal_hits['pos_z']      = ev_ch_pos_z[valid]

                n_hits = len(p_arr)
                event_key = f'{count}.{file_counter}'

                feature_dump = pd.DataFrame({
                    'track_index'  : dedup_track_idx,
                    'hit_id'       : hit_id_strings,
                    'x_position'   : x_arr,
                    'y_position'   : y_arr,
                    'z_position'   : z_arr,
                    'hit_rho'      : rho_arr,
                    'hit_position' : r_arr,
                    'edep'         : e_arr,
                    't_diff'       : t_diff_arr,
                    'true_mom_x'   : np.full(n_hits, t_px),
                    'true_mom_y'   : np.full(n_hits, t_py),
                    'true_mom_z'   : np.full(n_hits, t_pz),
                    # Calorimeter cluster — same scalar repeated for every hit row
                    'calo_edep'      : np.full(n_hits, c_edep),
                    'calo_edeperr'   : np.full(n_hits, c_edeperr),
                    'calo_ctime'     : np.full(n_hits, c_ctime),
                    'calo_ctimeerr'  : np.full(n_hits, c_ctimeerr),
                    'calo_did'       : np.full(n_hits, c_did),
                    'calo_doca'      : np.full(n_hits, c_doca),
                    'calo_dt'        : np.full(n_hits, c_dt),
                    'calo_dphidot'   : np.full(n_hits, c_dphidot),
                    'calo_ptoca'     : np.full(n_hits, c_ptoca),
                    'calo_tocavar'   : np.full(n_hits, c_tocavar),
                    'calo_tresid'    : np.full(n_hits, c_tresid),
                    'calo_tresidmvar': np.full(n_hits, c_tresidmvar),
                    'calo_tresidpvar': np.full(n_hits, c_tresidpvar),
                    'calo_cdepth'    : np.full(n_hits, c_cdepth),
                    'calo_trkdepth'  : np.full(n_hits, c_trkdepth),
                    'calo_csize'     : np.full(n_hits, c_csize),
                    'calo_poca_x'    : np.full(n_hits, c_poca_x),
                    'calo_poca_y'    : np.full(n_hits, c_poca_y),
                    'calo_poca_z'    : np.full(n_hits, c_poca_z),
                    'calo_mom_x'     : np.full(n_hits, c_mom_x),
                    'calo_mom_y'     : np.full(n_hits, c_mom_y),
                    'calo_mom_z'     : np.full(n_hits, c_mom_z),
                    'calo_matched'   : np.full(n_hits, c_matched),
                })

                feature_dump['event_index'] = event_key
                feature_variables.append(feature_dump)

                # Store crystal hits keyed by (event_key, dedup_track_idx)
                calo_hits_dict[(event_key, dedup_track_idx)] = crystal_hits

    if not feature_variables:
        print('\n\033[91mNo tracks matched the filtering criteria across files.\033[0m')
        return pd.DataFrame(), {}

    if duplicate_tracks_skipped > 0:
        print(f'\nSkipped {duplicate_tracks_skipped} duplicate track(s) during extraction.')

    events_df = pd.concat(feature_variables, ignore_index=True)

    print('\n\033[92mAll file data has been successfully loaded into the flat DataFrame.\033[0m')

    return events_df, calo_hits_dict


if __name__ == '__main__':
    import glob
    import sys

    ROOT_DIR = '/exp/mu2e/app/users/dgmyers/MLWork_EAF/DOA/MLData/rootFiles'

    if len(sys.argv) > 1:
        ROOT_DIR = sys.argv[1]

    root_files = sorted(glob.glob(os.path.join(ROOT_DIR, '*.root')))

    if not root_files:
        print(f'\033[91mNo .root files found in {ROOT_DIR}\033[0m')
        sys.exit(1)

    print(f'Found {len(root_files)} ROOT file(s) in {ROOT_DIR}:')
    for f in root_files:
        print(f'  {f}')

    filepaths_string = '\n'.join(root_files)
    df, calo_dict = uproot_data_extractor(filepaths_string)

    if not df.empty:
        n_tracks = df.groupby(['event_index', 'track_index']).ngroups
        print("\n=== Data Extraction Summary ===")
        print(f"Total Hits:   {df.shape[0]:,}")
        print(f"Total Tracks: {n_tracks:,}")
        print(f"Columns:      {list(df.columns)}")
        print("\n--- Sample rows (tracker + calo columns) ---")
        calo_cols = ['event_index', 'track_index', 'hit_id',
                     'true_mom_x', 'true_mom_y', 'true_mom_z',
                     'calo_matched', 'calo_edep', 'calo_ctime', 'calo_did']
        print(df[calo_cols].head(20))
        matched = df.groupby(['event_index', 'track_index'])['calo_matched'].first()
        print(f"\nTracks with calo match:    {int(matched.sum()):,} ({100*matched.mean():.1f}%)")
        print(f"Tracks without calo match: {int((~matched).sum()):,} ({100*(1-matched.mean()):.1f}%)")

        # Show a sample of crystal hits
        n_with_crystals = sum(1 for v in calo_dict.values() if len(v) > 0)
        print(f"\nTracks with crystal hits resolved: {n_with_crystals:,}")
        sample_key = next((k for k, v in calo_dict.items() if len(v) > 0), None)
        if sample_key is not None:
            print(f"\nSample crystal hits for track {sample_key}:")
            arr = calo_dict[sample_key]
            for row in arr:
                print(f"  crystalId={row['crystal_id']:5d}  eDep={row['edep']:.4f} MeV  "
                      f"time={row['time']:.2f} ns  "
                      f"pos=({row['pos_x']:.1f}, {row['pos_y']:.1f}, {row['pos_z']:.1f}) mm")
    else:
        print("\nExtraction failed or returned an empty DataFrame.")
