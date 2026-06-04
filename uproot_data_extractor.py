import uproot
import awkward as ak
import numpy as np
import pandas as pd
import os
import hashlib

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
    Extracts Mu2e tracker hit data directly from a long string of file paths,
    properly flattening track-level and hit-level nested dimensions.
    """
    feature_variables = []
    file_counter = 0
    duplicate_tracks_skipped = 0
    
    file_list = filepaths_string.strip().split()
    if not file_list:
        print("\n\033[91mError: Provided file paths string is empty!\033[0m")
        return pd.DataFrame()

    # Leaves to load 
    branches_to_load = [
        "trkhits",
        "trkmc.nhits",
        "trkmcsim" 
    ]
    
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
                trkana = tree.arrays(expressions=branches_to_load)
                
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
        r_hit_targets = np.sqrt(xpos_targets ** 2 + ypos_targets ** 2 + zpos_targets ** 2)
                
        endtime = hits['etime']
        edep = hits['edep']

        # --- Extract Track-Level MC True INITIAL Momentum Components ---
        mcsim = trkana["trkmcsim"]
        
        mom_x = mcsim['mom']['fCoordinates']['fX']
        mom_y = mcsim['mom']['fCoordinates']['fY']
        mom_z = mcsim['mom']['fCoordinates']['fZ']

        # --- Build the Flat DataFrames ---
        for count in range(len(xpos_targets)):
            num_tracks = len(plane[count])
            seen_track_fps = set()
            
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
                t0_arr = time_entry[:, 0]  
                t1_arr = time_entry[:, 1]  
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

                # Extract the primary matched MC particle (index 0) vector components
                track_mc_px = mom_x[count][track_idx]
                track_mc_py = mom_y[count][track_idx]
                track_mc_pz = mom_z[count][track_idx]

                if len(track_mc_px) > 0:
                    t_px = track_mc_px[0]
                    t_py = track_mc_py[0]
                    t_pz = track_mc_pz[0]
                else:
                    t_px = np.nan 
                    t_py = np.nan
                    t_pz = np.nan
                
                mom_x_arr = np.full(len(p_arr), t_px)
                mom_y_arr = np.full(len(p_arr), t_py)
                mom_z_arr = np.full(len(p_arr), t_pz)

                feature_dump = pd.DataFrame({
                    'track_index': dedup_track_idx,       
                    'hit_id'     : hit_id_strings,  
                    'x_position' : x_arr,           
                    'y_position' : y_arr,           
                    'z_position' : z_arr,           
                    'hit_rho'    : rho_arr,         
                    'hit_position': r_arr,          
                    'edep'       : e_arr,           
                    't_diff'     : t_diff_arr,      
                    'true_mom_x' : mom_x_arr,
                    'true_mom_y' : mom_y_arr,
                    'true_mom_z' : mom_z_arr       
                })
                
                feature_dump['event_index'] = f'{count}.{file_counter}'
                feature_variables.append(feature_dump)

    if not feature_variables:
        print('\n\033[91mNo tracks matched the filtering criteria across files.\033[0m')
        return pd.DataFrame()

    if duplicate_tracks_skipped > 0:
        print(f'\nSkipped {duplicate_tracks_skipped} duplicate track(s) during extraction.')

    events_df = pd.concat(feature_variables, ignore_index=True)
    
    print('\n\033[92mAll file data has been successfully loaded into the flat DataFrame.\033[0m')

    return events_df

if __name__ == '__main__':
    
    my_long_filepaths_string = """
    /exp/mu2e/data/users/dgmyers/MLData/nts.mu2e.CePlusEndpointOnSpill-reco-ntuple.MDC2025-002.001430_00001568.root
    """
    
    df = uproot_data_extractor(my_long_filepaths_string)
    
    if not df.empty:
        print("\n=== Data Extraction Verification ===")
        print(f"Total Rows (Hits) Extracted: {df.shape[0]}")
        print(f"DataFrame Columns: {list(df.columns)}")
        print("\n--- Rows of the DataFrame ---")
        print(df[['event_index', 'track_index', 'hit_id', 'true_mom_x', 'true_mom_y', 'true_mom_z']].head(15))
    else:
        print("\nExtraction failed or returned an empty DataFrame.")