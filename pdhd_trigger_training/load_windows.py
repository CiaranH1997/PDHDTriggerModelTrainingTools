import uproot
import awkward as ak
import numpy as np
import pandas as pd
import h5py
import ROOT # type: ignore

# Define functions for processing data from intput root files

'''
Create dictionary of TPs indows on each APA plane for each event with the following structure
{ eventID:
    { APA: { window_index: [ {"time": ..., "channel": ..., "charge": ...}, ... ],
             ... },
      ... }
}
'''

def read_tp_data(filename, include_broken_apa=False):
    # Open the ROOT file.
    file = uproot.open(filename)
    
    # Define a mapping from APA labels to tree names.
    apa_trees = {
        "APA1": "NP04TriggerTrainingAndAnalysis/TPWindowAPA1Tree",
        "APA2": "NP04TriggerTrainingAndAnalysis/TPWindowAPA2Tree",
        "APA3": "NP04TriggerTrainingAndAnalysis/TPWindowAPA3Tree",
        "APA4": "NP04TriggerTrainingAndAnalysis/TPWindowAPA4Tree"
    }
    
    apa_timebranch = {
        "APA1": "APA1Window_timepeak",
        "APA2": "APA2Window_timepeak",
        "APA3": "APA3Window_timepeak",
        "APA4": "APA4Window_timepeak"
    }
    apa_chanbranch = {
        "APA1": "APA1Window_channelid",
        "APA2": "APA2Window_channelid",
        "APA3": "APA3Window_channelid",
        "APA4": "APA4Window_channelid"
    }
    apa_adcbranch = {
        "APA1": "APA1Window_adcintegral",
        "APA2": "APA2Window_adcintegral",
        "APA3": "APA3Window_adcintegral",
        "APA4": "APA4Window_adcintegral"
    }
    apa_totbranch = {
        "APA1": "APA1Window_tot",
        "APA2": "APA2Window_tot",
        "APA3": "APA3Window_tot",
        "APA4": "APA4Window_tot"
    }
    apa_adcpeakbranch = {
        "APA1": "APA1Window_adcpeak",
        "APA2": "APA2Window_adcpeak",
        "APA3": "APA3Window_adcpeak",
        "APA4": "APA4Window_adcpeak"
    }

    if (not include_broken_apa):
        # Exclude APA1 if broken APA data is not to be included
        apa_trees.pop("APA1")
        apa_timebranch.pop("APA1")
        apa_chanbranch.pop("APA1")
        apa_adcbranch.pop("APA1")
        apa_totbranch.pop("APA1")
        apa_adcpeakbranch.pop("APA1")
    
    # This will hold the final nested dictionary.
    results = {}
    
    # Loop over each APA tree.
    for apa, tree_name in apa_trees.items():
        tree = file[tree_name]
        
        # Read the branches as awkward arrays.
        event_ids = tree["EventIterator"].array(library="ak")
        times     = tree[apa_timebranch[apa]].array(library="ak")
        channels  = tree[apa_chanbranch[apa]].array(library="ak")
        charges   = tree[apa_adcbranch[apa]].array(library="ak")
        tot       = tree[apa_totbranch[apa]].array(library="ak")
        adcpeak   = tree[apa_adcpeakbranch[apa]].array(library="ak")
        # Convert awkward arrays to lists.
        
        # Loop over events in this tree.
        for i, evt in enumerate(event_ids):
        # for i, evt in enumerate(tree):
            #event_id = i
            event_id = int(evt)
            # Initialize the dictionary for this event if not already done.
            if event_id not in results:
                results[event_id] = {}
            # Initialize the APA dictionary.
            if apa not in results[event_id]:
                results[event_id][apa] = {}
            
            # Convert the awkward arrays to Python lists.
            # Each of these is expected to be a list of 10 windows.
            windows_times    = ak.to_list(times[i])
            windows_channels = ak.to_list(channels[i])
            windows_charges  = ak.to_list(charges[i])
            windows_tot      = ak.to_list(tot[i])
            windows_adcpeak  = ak.to_list(adcpeak[i])
            
            # Loop over the 10 windows.
            #for win_idx, (win_time, win_channel, win_charge, windows_tot, windows_adcpeak) in enumerate(
            #    zip(windows_times, windows_channels, windows_charges, windows_tot, windows_adcpeak)
            #):
            #    tp_list = []
                # Each win_time, win_channel, win_charge is a list of integers.
            #    for t, ch, cq, tt, cp in zip(win_time, win_channel, win_charge, windows_tot, windows_adcpeak):
            #        tp_list.append({"Time_peak": t, "ChannelID": ch, "ADC_integral": cq, "ToT": tt, "ADC_peak": cp})
            #    results[event_id][apa][win_idx] = tp_list

            for win_idx, (win_time, win_channel, win_charge, win_tot, win_adcpeak) in enumerate(
                zip(windows_times, windows_channels, windows_charges, windows_tot, windows_adcpeak)
                ):
                tp_list = []
                # Each win_time, win_channel, win_charge is a list of integers.
                for t, ch, cq, tt, cp in zip(win_time, win_channel, win_charge, win_tot, win_adcpeak):
                    tp_list.append({"Time_peak": t, "ChannelID": ch, "ADC_integral": cq, "ToT": tt, "ADC_peak": cp})
                results[event_id][apa][win_idx] = tp_list

    print(f'Total events processed: {len(results)}')
    return results

def read_neutrino_tp_data(filename, include_broken_apa=False):
    # Open the ROOT file.
    file = uproot.open(filename)
    
    # --- Step 1. Build a mapping from event id to neutrino APA using GenieTruth ---
    gt_tree = file["NP04TriggerTrainingAndAnalysis/GenieTruth"]
    gt_event_ids = gt_tree["EventIterator"].array(library="ak")
    gt_APAs = gt_tree["APA"].array(library="ak")
    
    # Create a mapping: event_id -> APA (integer)
    nu_truth = {}
    for eventid, apa in zip(ak.to_list(gt_event_ids), ak.to_list(gt_APAs)):
        nu_truth[int(eventid)] = int(apa)
    
    tree_name = "NP04TriggerTrainingAndAnalysis/TPNuWindowTree"
    
    apa_timebranch = "NuWindow_timepeak"
    apa_chanbranch = "NuWindow_channelid"
    apa_adcbranch  = "NuWindow_adcintegral"
    apa_totbranch  = "NuWindow_tot"
    apa_adcpeakbranch = "NuWindow_adcpeak"
    
    # This will hold the final nested dictionary.
    results = {}
        
    # Open the neutrino TP tree for the current APA.
    tree = file[tree_name]
        
    # Read the branches as awkward arrays.
    event_ids = tree["EventIterator"].array(library="ak")
    apa = tree["APA"].array(library="ak")
    times     = tree[apa_timebranch].array(library="ak")
    channels  = tree[apa_chanbranch].array(library="ak")
    charges   = tree[apa_adcbranch].array(library="ak")
    tot       = tree[apa_totbranch].array(library="ak")
    adcpeak   = tree[apa_adcpeakbranch].array(library="ak")
        
    # Convert awkward arrays to lists.
    event_ids_list    = ak.to_list(event_ids)
    apa_list          = ak.to_list(apa)
    windows_times     = ak.to_list(times)
    windows_channels  = ak.to_list(channels)
    windows_charges   = ak.to_list(charges)
    windows_tot       = ak.to_list(tot)
    windows_adcpeak   = ak.to_list(adcpeak)

    print(f'num events = {len(event_ids_list)}')
    print(f'window times length = {len(windows_times)}')

    # Process each event in this tree.
    for i, evt in enumerate(event_ids_list):
    #for i, evt in enumerate(tree):
        #event_id = i
        event_id = int(evt)  # Ensure event_id is non-negative and avoid unnecessary adjustments
                
        #print(f'APA = {apa[event_id]}')
        if len(windows_times) == 0:
            current_apa = apa[i] if i < len(apa) else "Unknown"
            print(f'Window has no TPs! Event ID: {event_id}, APA: {current_apa}')
            continue  # No window data for this event.
              
        if (not include_broken_apa) and (apa_list[event_id]==1):
            # Skip broken APA1 data unless specified otherwise.
            continue

        # Initialize the event dictionary if needed.
        if event_id not in results:
            results[event_id] = {}
        # Use the tree name as key; you could also use a label like f"APA{apa}"
        apa_key = f"APA{apa_list[event_id]}"
        #print(f'APA key = {apa_key}')
        if apa_key not in results[event_id]:
            results[event_id][apa_key] = {}
        
        # For this event, get the list of windows (should be 10 windows).
        event_windows_times    = windows_times[i]
        event_windows_channels = windows_channels[i]
        event_windows_charges  = windows_charges[i]
        event_windows_tot      = windows_tot[i]
        event_windows_adcpeak  = windows_adcpeak[i]
        
        tp_list = []
        
        for t, ch, cq, tt, cp in zip(event_windows_times[0], event_windows_channels[0], event_windows_charges[0], event_windows_tot[0], event_windows_adcpeak[0]):
            tp_list.append({"Time_peak": t, "ChannelID": ch, "ADC_integral": cq, "ToT": tt, "ADC_peak": cp})
            results[event_id][apa_key][0] = tp_list
        


    print(f'Total events processed: {len(results)}')
    return results


def read_tp_data_to_hdf5(filename, output_hdf5, include_broken_apa=False):
    # Open the ROOT file.
    file = uproot.open(filename)
    
    # Mapping from APA labels to tree names.
    apa_trees = {
        "APA1": "NP04TriggerTrainingAndAnalysis/TPWindowAPA1Tree",
        "APA2": "NP04TriggerTrainingAndAnalysis/TPWindowAPA2Tree",
        "APA3": "NP04TriggerTrainingAndAnalysis/TPWindowAPA3Tree",
        "APA4": "NP04TriggerTrainingAndAnalysis/TPWindowAPA4Tree"
    }
    
    branch_names = {
        "time":   {"APA1": "APA1Window_timepeak",   "APA2": "APA2Window_timepeak",   "APA3": "APA3Window_timepeak",   "APA4": "APA4Window_timepeak"},
        "chan":   {"APA1": "APA1Window_channelid",  "APA2": "APA2Window_channelid",  "APA3": "APA3Window_channelid",  "APA4": "APA4Window_channelid"},
        "adcint": {"APA1": "APA1Window_adcintegral","APA2": "APA2Window_adcintegral","APA3": "APA3Window_adcintegral","APA4": "APA4Window_adcintegral"},
        "tot":    {"APA1": "APA1Window_tot",        "APA2": "APA2Window_tot",        "APA3": "APA3Window_tot",        "APA4": "APA4Window_tot"},
        "adcpeak":{"APA1": "APA1Window_adcpeak",    "APA2": "APA2Window_adcpeak",    "APA3": "APA3Window_adcpeak",    "APA4": "APA4Window_adcpeak"},
    }

    if not include_broken_apa:
        for k in list(apa_trees.keys()):
            if k == "APA1":
                for b in branch_names.values():
                    b.pop(k, None)
                apa_trees.pop(k, None)
    
    # Initialize event ID counter
    event_id_counter = 0
    # Create output HDF5 file
    with h5py.File(output_hdf5, "w") as h5file:
        event_counter = 0

        # Loop over each APA tree
        for apa, tree_name in apa_trees.items():
            tree = file[tree_name]
            
            event_ids = tree["EventIterator"].array(library="ak")
            times     = tree[branch_names["time"][apa]].array(library="ak")
            channels  = tree[branch_names["chan"][apa]].array(library="ak")
            charges   = tree[branch_names["adcint"][apa]].array(library="ak")
            tot       = tree[branch_names["tot"][apa]].array(library="ak")
            adcpeak   = tree[branch_names["adcpeak"][apa]].array(library="ak")
            
            for i, evt in enumerate(event_ids):
                #event_id = int(evt)  # Use actual EventIterator value
                event_id = event_id_counter
                event_id_counter += 1
                
                # Create or get event group
                event_grp_name = f"event_{event_id}"
                event_grp = h5file.require_group(event_grp_name)
                
                # Create APA group
                apa_grp = event_grp.require_group(apa)
                
                # Convert awkward to python lists
                win_times    = ak.to_list(times[i])
                win_channels = ak.to_list(channels[i])
                win_charges  = ak.to_list(charges[i])
                win_tot      = ak.to_list(tot[i])
                win_adcpeak  = ak.to_list(adcpeak[i])
                
                for win_idx, (t_list, ch_list, cq_list, tt_list, cp_list) in enumerate(
                    zip(win_times, win_channels, win_charges, win_tot, win_adcpeak)
                ):
                    if len(t_list) == 0:
                        continue

                    win_grp = apa_grp.require_group(f"window_{win_idx}")
                    win_grp.create_dataset("time", data=np.array(t_list, dtype=np.int32))
                    win_grp.create_dataset("channel", data=np.array(ch_list, dtype=np.int32))
                    win_grp.create_dataset("adc_integral", data=np.array(cq_list, dtype=np.float32))
                    win_grp.create_dataset("tot", data=np.array(tt_list, dtype=np.float32))
                    win_grp.create_dataset("adc_peak", data=np.array(cp_list, dtype=np.float32))

                event_counter += 1

        print(f"[HDF5] Finished writing {event_counter} events to {output_hdf5}")


def read_tp_data_to_hdf5_iterate(filename, output_hdf5, include_broken_apa=False, step_size=1000):
    """
    Stream TP data from ROOT into HDF5 incrementally, one APA at a time,
    using uproot.iterate to avoid loading everything into memory.
    """
    # --- Mapping from APA labels to tree names ---
    apa_trees = {
        "APA1": "NP04TriggerTrainingAndAnalysis/TPWindowAPA1Tree",
        "APA2": "NP04TriggerTrainingAndAnalysis/TPWindowAPA2Tree",
        "APA3": "NP04TriggerTrainingAndAnalysis/TPWindowAPA3Tree",
        "APA4": "NP04TriggerTrainingAndAnalysis/TPWindowAPA4Tree",
    }

    # --- Branch names ---
    branch_names = {
        "time":   {"APA1": "APA1Window_timepeak",   "APA2": "APA2Window_timepeak",   "APA3": "APA3Window_timepeak",   "APA4": "APA4Window_timepeak"},
        "chan":   {"APA1": "APA1Window_channelid",  "APA2": "APA2Window_channelid",  "APA3": "APA3Window_channelid",  "APA4": "APA4Window_channelid"},
        "adcint": {"APA1": "APA1Window_adcintegral","APA2": "APA2Window_adcintegral","APA3": "APA3Window_adcintegral","APA4": "APA4Window_adcintegral"},
        "tot":    {"APA1": "APA1Window_tot",        "APA2": "APA2Window_tot",        "APA3": "APA3Window_tot",        "APA4": "APA4Window_tot"},
        "adcpeak":{"APA1": "APA1Window_adcpeak",    "APA2": "APA2Window_adcpeak",    "APA3": "APA3Window_adcpeak",    "APA4": "APA4Window_adcpeak"},
    }

    # Optionally drop APA1 if broken
    if not include_broken_apa:
        apa_trees.pop("APA1", None)
        for b in branch_names.values():
            b.pop("APA1", None)

    # --- Initialize event ID counter ---
    event_id_counter = 0
    # --- Open HDF5 file ---
    with h5py.File(output_hdf5, "w") as h5file:

        # --- Loop over each APA tree separately ---
        for apa, tree_name in apa_trees.items():
            print(f"[HDF5] Processing {apa} from {tree_name}")

            # Branch list to read per chunk
            branches = [
                "EventIterator",
                branch_names["time"][apa],
                branch_names["chan"][apa],
                branch_names["adcint"][apa],
                branch_names["tot"][apa],
                branch_names["adcpeak"][apa],
            ]

            # Stream through the tree in chunks
            for arrays in uproot.iterate(
                f"{filename}:{tree_name}",
                expressions=branches,
                step_size=step_size,
                library="ak",
            ):
                # Extract fields for this chunk
                event_ids = ak.to_list(arrays["EventIterator"])
                times_arr = ak.to_list(arrays[branch_names["time"][apa]])
                chans_arr = ak.to_list(arrays[branch_names["chan"][apa]])
                adcint_arr = ak.to_list(arrays[branch_names["adcint"][apa]])
                tot_arr = ak.to_list(arrays[branch_names["tot"][apa]])
                adcpeak_arr = ak.to_list(arrays[branch_names["adcpeak"][apa]])

                # Loop over events in this chunk
                for i, evt in enumerate(event_ids):
                    #event_id = int(evt)
                    event_id = event_id_counter
                    event_id_counter += 1
                    event_grp_name = f"event_{event_id}"

                    # Create or get event group
                    event_grp = h5file.require_group(event_grp_name)
                    apa_grp = event_grp.require_group(apa)

                    # Get windowed TP lists
                    win_times = times_arr[i]
                    win_channels = chans_arr[i]
                    win_charges = adcint_arr[i]
                    win_tot = tot_arr[i]
                    win_adcpeak = adcpeak_arr[i]

                    # Loop over each window in the event
                    for win_idx, (t_list, ch_list, cq_list, tt_list, cp_list) in enumerate(
                        zip(win_times, win_channels, win_charges, win_tot, win_adcpeak)
                    ):
                        if len(t_list) == 0:
                            continue

                        win_grp = apa_grp.require_group(f"window_{win_idx}")
                        win_grp.create_dataset("time", data=np.array(t_list, dtype=np.int32), compression="gzip")
                        win_grp.create_dataset("channel", data=np.array(ch_list, dtype=np.int32), compression="gzip")
                        win_grp.create_dataset("adc_integral", data=np.array(cq_list, dtype=np.float32), compression="gzip")
                        win_grp.create_dataset("tot", data=np.array(tt_list, dtype=np.float32), compression="gzip")
                        win_grp.create_dataset("adc_peak", data=np.array(cp_list, dtype=np.float32), compression="gzip")

            print(f"[HDF5] Finished streaming {apa}")

    print(f"[HDF5] ✅ Finished writing all APAs to {output_hdf5}")
    print(f"[HDF5] Total event-APA-window groups written: {event_id_counter}")


def read_neutrino_tp_data_to_hdf5(filename, hdf5_outfile, include_broken_apa=False):
    """
    Read neutrino TP data from a ROOT file and write it out incrementally
    to an HDF5 file. Avoids keeping the full dataset in memory.
    """
    # --- Open ROOT file ---
    file = uproot.open(filename)

    # --- Step 1. Build mapping from event_id to neutrino APA (GenieTruth) ---
    gt_tree = file["NP04TriggerTrainingAndAnalysis/GenieTruth"]
    gt_event_ids = gt_tree["EventIterator"].array(library="ak")
    gt_APAs = gt_tree["APA"].array(library="ak")
    
    nu_truth = {int(eid): int(apa) for eid, apa in zip(ak.to_list(gt_event_ids), ak.to_list(gt_APAs))}

    # --- Step 2. Load neutrino TP tree ---
    tree_name = "NP04TriggerTrainingAndAnalysis/TPNuWindowTree"
    tree = file[tree_name]
    
    # Branch names
    apa_branch = "APA"
    branches = {
        "time": "NuWindow_timepeak",
        "channel": "NuWindow_channelid",
        "adc_integral": "NuWindow_adcintegral",
        "tot": "NuWindow_tot",
        "adc_peak": "NuWindow_adcpeak"
    }

    # Read branches
    event_ids = tree["EventIterator"].array(library="ak")
    apa_array = tree[apa_branch].array(library="ak")
    arrays = {name: tree[branch].array(library="ak") for name, branch in branches.items()}

    # Convert to lists for easy indexing
    event_ids_list = ak.to_list(event_ids)
    apa_list = ak.to_list(apa_array)
    arrays_list = {k: ak.to_list(v) for k, v in arrays.items()}

    print(f"Number of neutrino events: {len(event_ids_list)}")

    event_id_counter = 0
    # --- Step 3. Create HDF5 file and write incrementally ---
    with h5py.File(hdf5_outfile, "w") as h5f:
        for i, evt in enumerate(event_ids_list):
            #event_id = int(evt)  # global unique index per entry
            event_id = event_id_counter
            event_id_counter += 1
            apa_val = apa_list[i]

            # Skip broken APA1 if requested
            if (not include_broken_apa) and apa_val == 1:
                continue

            # Check that there is data
            if len(arrays_list["time"]) == 0 or i >= len(arrays_list["time"]):
                print(f"Skipping event {event_id} (no TPs)")
                continue

            apa_key = f"APA{apa_val}"
            event_group = h5f.require_group(str(event_id))
            apa_group = event_group.require_group(apa_key)

            # Extract the "window 0" TPs for this event
            times = np.array(arrays_list["time"][i][0])
            channels = np.array(arrays_list["channel"][i][0])
            charges = np.array(arrays_list["adc_integral"][i][0])
            tots = np.array(arrays_list["tot"][i][0])
            adc_peaks = np.array(arrays_list["adc_peak"][i][0])

            # Write datasets for this window
            apa_group.create_dataset("0_time", data=times, compression="gzip")
            apa_group.create_dataset("0_channel", data=channels, compression="gzip")
            apa_group.create_dataset("0_charge", data=charges, compression="gzip")
            apa_group.create_dataset("0_tot", data=tots, compression="gzip")
            apa_group.create_dataset("0_adcpeak", data=adc_peaks, compression="gzip")

    print(f"✅ HDF5 file written to: {hdf5_outfile}")


def read_neutrino_tp_data_to_hdf5_iterate(filename, hdf5_outfile, include_broken_apa=False, step_size=1000):
    """
    Stream neutrino TP data from a ROOT file and write incrementally to HDF5.
    Uses uproot.iterate to avoid loading all events into RAM.
    """
    # --- Step 1. Build mapping from event_id to neutrino APA (GenieTruth) ---
    file = uproot.open(filename)
    gt_tree = file["NP04TriggerTrainingAndAnalysis/GenieTruth"]
    gt_event_ids = gt_tree["EventIterator"].array(library="ak")
    gt_APAs = gt_tree["APA"].array(library="ak")
    nu_truth = {int(eid): int(apa) for eid, apa in zip(ak.to_list(gt_event_ids), ak.to_list(gt_APAs))}
    del gt_tree, gt_event_ids, gt_APAs  # free memory

    # --- Step 2. Define tree and branches ---
    tree_name = "NP04TriggerTrainingAndAnalysis/TPNuWindowTree"
    apa_branch = "APA"
    branch_map = {
        "time": "NuWindow_timepeak",
        "channel": "NuWindow_channelid",
        "adc_integral": "NuWindow_adcintegral",
        "tot": "NuWindow_tot",
        "adc_peak": "NuWindow_adcpeak"
    }

    branches = ["EventIterator", apa_branch] + list(branch_map.values())

    # --- Step 3. Stream through the neutrino TP tree ---
    event_id_counter = 0
    with h5py.File(hdf5_outfile, "w") as h5f:
        for arrays in uproot.iterate(
            f"{filename}:{tree_name}",
            expressions=branches,
            step_size=step_size,
            library="ak",
        ):
            # Convert the fields we need for this chunk
            event_ids = ak.to_list(arrays["EventIterator"])
            apa_vals = ak.to_list(arrays[apa_branch])

            # Convert the TP fields
            time_chunk = ak.to_list(arrays[branch_map["time"]])
            chan_chunk = ak.to_list(arrays[branch_map["channel"]])
            adcint_chunk = ak.to_list(arrays[branch_map["adc_integral"]])
            tot_chunk = ak.to_list(arrays[branch_map["tot"]])
            adcpeak_chunk = ak.to_list(arrays[branch_map["adc_peak"]])

            # Loop through events in this chunk
            for i, evt in enumerate(event_ids):
                #event_id = int(evt)
                event_id = event_id_counter
                event_id_counter += 1
                apa_val = int(apa_vals[i])

                # Skip broken APA1 if required
                if (not include_broken_apa) and apa_val == 1:
                    continue

                # Handle events with no TPs
                if i >= len(time_chunk) or len(time_chunk[i]) == 0:
                    continue

                # Create group structure in HDF5
                apa_key = f"APA{apa_val}"
                event_group = h5f.require_group(str(event_id))
                apa_group = event_group.require_group(apa_key)

                # For neutrino tree, we expect a single window (index 0)
                tps_time = np.array(time_chunk[i][0], dtype=np.int32)
                tps_chan = np.array(chan_chunk[i][0], dtype=np.int32)
                tps_adcint = np.array(adcint_chunk[i][0], dtype=np.float32)
                tps_tot = np.array(tot_chunk[i][0], dtype=np.float32)
                tps_adcpeak = np.array(adcpeak_chunk[i][0], dtype=np.float32)

                # Write datasets with compression
                apa_group.create_dataset("time", data=tps_time, compression="gzip")
                apa_group.create_dataset("channel", data=tps_chan, compression="gzip")
                apa_group.create_dataset("adc_integral", data=tps_adcint, compression="gzip")
                apa_group.create_dataset("tot", data=tps_tot, compression="gzip")
                apa_group.create_dataset("adc_peak", data=tps_adcpeak, compression="gzip")

    print(f"[HDF5] ✅ Finished writing neutrino TP data to {hdf5_outfile}")
    print(f"[HDF5] Total neutrino events written: {event_id_counter}")


# Function to bin the TPs in a time series array
def bin_windows_by_time(sub_grouped_data, 
                        bin_width=1000, 
                        window_length=20000, 
                        adc_integral=True,
                        tot=False,
                        adc_peak=False,
                        mean=False):                
    if tot or adc_peak:
        adc_integral = False  # If ToT or ADC_peak is used, do not use ADC_integral
    if tot and adc_peak:
        raise ValueError("Cannot use both ToT and ADC_peak together. Choose one or neither.")
    
    binned_data = {}

    for event_id in sub_grouped_data:
        binned_data[event_id] = {}

        for apa in sub_grouped_data[event_id]:
            binned_data[event_id][apa] = []  # Store all sub-events as lists
            #for sub_event in sub_grouped_data[event_id][apa]:
            for i, window in enumerate(sub_grouped_data[event_id][apa]):
                
                #time_data = np.array(window["Time_peak"])
                tp_list = sub_grouped_data[event_id][apa][i]
                time_data = np.array([tp["Time_peak"] for tp in tp_list])
        
                if len(time_data) == 0:
                    continue
        
                # Could enforce this and remove events that are do not have 20 elements?
                #time_min = time_data.min()
                #time_max = time_data.max()
        
                time_min = time_data.min()
                time_max = time_min + window_length
                
                df = pd.DataFrame(tp_list)

                # Define bin edges (0 to window_length with step = bin_width)
                bins = np.arange(time_min, time_max + bin_width, bin_width)

                # Convert awkward array to NumPy
                df["Time_peak"] = ak.to_numpy(df["Time_peak"])

                # Assign each TP to a time bin
                df["Time_bin"] = pd.cut(df["Time_peak"], bins=bins, labels=False, right=False)

                # Drop NaN values (TPs outside bin range)
                df = df.dropna(subset=["Time_bin"])
                df["Time_bin"] = df["Time_bin"].astype(int)  # Ensure it's an integer
       
                # Predefine binned_adc as an empty Series in case no condition is met
                binned_adc = pd.Series(dtype=float)
                # Sum ADC_Integral within each time bin
                if adc_integral:
                    if not mean:
                        binned_adc = df.groupby("Time_bin")["ADC_integral"].sum()
                    else:
                        binned_adc = df.groupby("Time_bin")["ADC_integral"].mean()
                elif tot:
                    if not mean:
                        binned_adc = df.groupby("Time_bin")["ToT"].sum()
                    else:
                        # If mean is True, calculate the mean of ToT in each bin
                        binned_adc = df.groupby("Time_bin")["ToT"].sum()
                elif adc_peak:
                    if not mean:
                        binned_adc = df.groupby("Time_bin")["ADC_peak"].sum()
                    else:
                        binned_adc = df.groupby("Time_bin")["ADC_peak"].sum()
                else:
                    binned_adc = df.groupby("Time_bin")["ADC_integral"].sum()

                # Convert to a NumPy array (handling missing bins with fillna)
                binned_array = binned_adc.reindex(range(len(bins) - 1), fill_value=0).to_numpy()

                binned_data[event_id][apa].append(binned_array)  # Store all sub-events

    return binned_data

def bin_windows_by_channel_and_time(sub_grouped_data, 
                                   time_bin_width=1000, 
                                   channel_bin_width=100, 
                                   window_length=20000,
                                   adc_integral=True,
                                   tot=False,
                                   adc_peak=False,
                                   mean=False):
    import numpy as np
    import pandas as pd
    import awkward as ak

    if tot or adc_peak:
        adc_integral = False  # If ToT or ADC_peak is used, do not use ADC_integral
    if tot and adc_peak:
        raise ValueError("Cannot use both ToT and ADC_peak together. Choose one or neither.")
    

    binned_data = {}

    for event_id in sub_grouped_data:
        binned_data[event_id] = {}

        for apa in sub_grouped_data[event_id]:
            binned_data[event_id][apa] = []

            for i, window in enumerate(sub_grouped_data[event_id][apa]):
                tp_list = sub_grouped_data[event_id][apa][i]
                
                if len(tp_list) == 0:
                    continue

                time_data = np.array([tp["Time_peak"] for tp in tp_list])
                channel_data = np.array([tp["ChannelID"] for tp in tp_list])

                time_min = time_data.min()
                time_max = time_min + window_length
                
                channel_min = channel_data.min()
                channel_max = channel_data.max()
                if apa == "APA1":
                    # APA1 has a different channel range
                    channel_min = 2080
                    channel_max = 2560
                elif apa == "APA2": 
                    channel_min = 7200
                    channel_max = 7680
                elif apa == "APA3":
                    channel_min = 4160
                    channel_max = 4640
                elif apa == "APA4":
                    channel_min = 9280
                    channel_max = 9760
                else:
                    raise ValueError(f"Unknown APA: {apa}. Expected one of 'APA1', 'APA2', 'APA3', 'APA4'.")
                
                df = pd.DataFrame(tp_list)

                # Define bin edges
                time_bins = np.arange(time_min, time_max + time_bin_width, time_bin_width)
                channel_bins = np.arange(channel_min, channel_max + channel_bin_width, channel_bin_width)

                df["Time_peak"] = ak.to_numpy(df["Time_peak"])
                df["ChannelID"] = ak.to_numpy(df["ChannelID"])
                df["ADC_integral"] = ak.to_numpy(df["ADC_integral"])
                df["ToT"] = ak.to_numpy(df["ToT"])
                df["ADC_peak"] = ak.to_numpy(df["ADC_peak"])

                # Bin assignment
                df["Time_bin"] = pd.cut(df["Time_peak"], bins=time_bins, labels=False, right=False)
                df["Channel_bin"] = pd.cut(df["ChannelID"], bins=channel_bins, labels=False, right=False)

                # Drop TPs outside the defined bin range
                df = df.dropna(subset=["Time_bin", "Channel_bin"])
                df["Time_bin"] = df["Time_bin"].astype(int)
                df["Channel_bin"] = df["Channel_bin"].astype(int)

                # 2D histogram: sum of ADC_integral in each (time, channel) bin
                hist_2d = np.zeros((len(time_bins) - 1, len(channel_bins) - 1), dtype=float)
                count_2d = np.zeros_like(hist_2d)

                for row in df.itertuples():
                    count_2d[row.Time_bin, row.Channel_bin] += 1
                    if adc_integral:
                        hist_2d[row.Time_bin, row.Channel_bin] += row.ADC_integral
                    elif tot:
                        hist_2d[row.Time_bin, row.Channel_bin] += row.ToT
                    elif adc_peak:
                        hist_2d[row.Time_bin, row.Channel_bin] += row.ADC_peak
                    else:
                        # Default case, should not happen if adc_integral, tot, or adc_peak is True
                        # This is just a safety check
                        print("Warning: No valid ADC data type selected. Using ADC_integral by default.")
                        hist_2d[row.Time_bin, row.Channel_bin] += row.ADC_integral
                # Append the 2D histogram for this sub-event
                if mean:
                    # If mean is True, normalize the histogram by the number of TPs in each bin
                    hist_2d = np.divide(hist_2d, count_2d, out=np.zeros_like(hist_2d), where=count_2d != 0)
                    #hist_2d /= df.groupby(["Time_bin", "Channel_bin"]).size().values.reshape(hist_2d.shape)
                
                binned_data[event_id][apa].append(hist_2d)

    return binned_data


'''
def bin_windows_by_channel_and_time_hdf5(hdf5_file,
                                         time_bin_width=1000,
                                         channel_bin_width=100,
                                         window_length=20000,
                                         adc_integral=True,
                                         tot=False,
                                         adc_peak=False,
                                         mean=False,
                                         max_events=None):
    """
    Bin TPs in an HDF5 file (APA or neutrino format) by channel and time.
    Returns a nested dictionary: event_id -> APA -> [2D histogram per window].
    """
    # Determine which ADC-like quantity to use
    if tot or adc_peak:
        adc_integral = False
    if tot and adc_peak:
        raise ValueError("Cannot use both ToT and ADC_peak together. Choose one or neither.")

    # APA-specific channel ranges
    apa_ranges = {
        "APA1": (2080, 2560),
        "APA2": (7200, 7680),
        "APA3": (4160, 4640),
        "APA4": (9280, 9760)
    }

    binned_data = {}

    with h5py.File(hdf5_file, "r") as h5f:
        event_keys = list(h5f.keys())
        if max_events is not None:
            event_keys = event_keys[:max_events]

        for ev_key in event_keys:
            # event_id = int(ev_key)
            #event_id = ev_key  # Keep as string key to match HDF5 structure
            if ev_key.startswith("event_"):
                event_id = int(ev_key.replace("event_", ""))
            else:
                event_id = int(ev_key)
            event_group = h5f[ev_key]
            binned_data[event_id] = {}

            for apa_key in event_group.keys():
                apa_group = event_group[apa_key]
                binned_data[event_id][apa_key] = []

                # --- Handle either multiple windows (APA cosmic format)
                #     or single "0_time"/"0_channel" etc (neutrino format)
                window_names = [w for w in apa_group.keys() if w.startswith("window_")]
                is_neutrino = False
                if len(window_names) == 0:
                    # Likely neutrino format
                    window_names = ["0"]
                    is_neutrino = True

                for win_name in window_names:
                    if is_neutrino:
                        # Neutrino format: datasets have "0_time", "0_channel", "0_charge", ...
                        try:
                            time_data = np.array(apa_group[f"{win_name}_time"][:])
                            channel_data = np.array(apa_group[f"{win_name}_channel"][:])
                        except KeyError:
                            continue

                        if adc_integral:
                            adc_data = np.array(apa_group[f"{win_name}_charge"][:])
                        elif tot:
                            adc_data = np.array(apa_group[f"{win_name}_tot"][:])
                        elif adc_peak:
                            adc_data = np.array(apa_group[f"{win_name}_adcpeak"][:])
                    else:
                        # APA cosmic format: nested groups like window_0/time, channel, etc
                        win_group = apa_group[win_name]
                        if "time" not in win_group:
                            continue

                        time_data = np.array(win_group["time"][:])
                        channel_data = np.array(win_group["channel"][:])
                        if adc_integral:
                            adc_data = np.array(win_group["adc_integral"][:])
                        elif tot:
                            adc_data = np.array(win_group["tot"][:])
                        elif adc_peak:
                            adc_data = np.array(win_group["adc_peak"][:])

                    if len(time_data) == 0 or len(channel_data) == 0:
                        continue

                    # Define bin ranges
                    time_min = time_data.min()
                    time_max = time_min + window_length

                    ch_min, ch_max = apa_ranges.get(apa_key, (channel_data.min(), channel_data.max()))
                    time_bins = np.arange(time_min, time_max + time_bin_width, time_bin_width)
                    channel_bins = np.arange(ch_min, ch_max + channel_bin_width, channel_bin_width)

                    # Bin assignment using pandas
                    df = pd.DataFrame({
                        "Time_peak": time_data,
                        "ChannelID": channel_data,
                        "Value": adc_data
                    })

                    df["Time_bin"] = pd.cut(df["Time_peak"], bins=time_bins, labels=False, right=False)
                    df["Channel_bin"] = pd.cut(df["ChannelID"], bins=channel_bins, labels=False, right=False)
                    df = df.dropna(subset=["Time_bin", "Channel_bin"])
                    df["Time_bin"] = df["Time_bin"].astype(int)
                    df["Channel_bin"] = df["Channel_bin"].astype(int)

                    hist_2d = np.zeros((len(time_bins)-1, len(channel_bins)-1), dtype=float)
                    count_2d = np.zeros_like(hist_2d)

                    for row in df.itertuples():
                        count_2d[row.Time_bin, row.Channel_bin] += 1
                        hist_2d[row.Time_bin, row.Channel_bin] += row.Value

                    if mean:
                        hist_2d = np.divide(hist_2d, count_2d,
                                            out=np.zeros_like(hist_2d),
                                            where=count_2d != 0)

                    binned_data[event_id][apa_key].append(hist_2d)

    return binned_data
'''
def bin_windows_by_channel_and_time_hdf5(hdf5_file,
                                         time_bin_width=1000,
                                         channel_bin_width=100,
                                         window_length=20000,
                                         max_events=None):
    """
    Bin TPs in an HDF5 file (APA or neutrino format) by channel and time.
    Returns a nested dictionary: event_id -> APA -> [2D histogram per window].
    """

    # APA-specific channel ranges
    apa_ranges = {
        "APA1": (2080, 2560),
        "APA2": (7200, 7680),
        "APA3": (4160, 4640),
        "APA4": (9280, 9760)
    }

    binned_data = {}

    with h5py.File(hdf5_file, "r") as h5f:
        event_keys = list(h5f.keys())
        if max_events is not None:
            event_keys = event_keys[:max_events]

        for ev_key in event_keys:
            # event_id = int(ev_key)
            #event_id = ev_key  # Keep as string key to match HDF5 structure
            if ev_key.startswith("event_"):
                event_id = int(ev_key.replace("event_", ""))
            else:
                event_id = int(ev_key)
            event_group = h5f[ev_key]
            binned_data[event_id] = {}

            for apa_key in event_group.keys():
                apa_group = event_group[apa_key]
                binned_data[event_id][apa_key] = []

                # Check for windowed or flat datasets
                if any(k.startswith("window_") for k in apa_group.keys()):
                    # APA cosmic format
                    window_names = [w for w in apa_group.keys() if w.startswith("window_")]
                    is_neutrino = False
                else:
                    # Neutrino format — no "window_" prefix, just flat datasets
                    window_names = ["0"]
                    is_neutrino = True

                for win_name in window_names:
                    if is_neutrino:
                        # Neutrino format: datasets are "time", "channel", etc.
                        try:
                            time_data = np.array(apa_group["time"][:])
                            channel_data = np.array(apa_group["channel"][:])
                        except KeyError:
                            continue

                        adc_data = np.array(apa_group["adc_integral"][:])
                    else:
                        # Cosmic format: window groups
                        win_group = apa_group[win_name]
                        if "time" not in win_group:
                            continue

                        time_data = np.array(win_group["time"][:])
                        channel_data = np.array(win_group["channel"][:])
                        adc_data = np.array(win_group["adc_integral"][:])
                        

                    if len(time_data) == 0 or len(channel_data) == 0:
                        continue

                    # Define bin ranges
                    time_min = time_data.min()
                    time_max = time_min + window_length

                    ch_min, ch_max = apa_ranges.get(apa_key, (channel_data.min(), channel_data.max()))
                    time_bins = np.arange(time_min, time_max + time_bin_width, time_bin_width)
                    channel_bins = np.arange(ch_min, ch_max + channel_bin_width, channel_bin_width)

                    # Bin assignment using pandas
                    df = pd.DataFrame({
                        "Time_peak": time_data,
                        "ChannelID": channel_data,
                        "Value": adc_data
                    })

                    df["Time_bin"] = pd.cut(df["Time_peak"], bins=time_bins, labels=False, right=False)
                    df["Channel_bin"] = pd.cut(df["ChannelID"], bins=channel_bins, labels=False, right=False)
                    df = df.dropna(subset=["Time_bin", "Channel_bin"])
                    df["Time_bin"] = df["Time_bin"].astype(int)
                    df["Channel_bin"] = df["Channel_bin"].astype(int)

                    hist_2d = np.zeros((len(time_bins)-1, len(channel_bins)-1), dtype=float)
                    count_2d = np.zeros_like(hist_2d)

                    for row in df.itertuples():
                        count_2d[row.Time_bin, row.Channel_bin] += 1
                        hist_2d[row.Time_bin, row.Channel_bin] += row.Value

                    binned_data[event_id][apa_key].append(hist_2d)

    return binned_data
