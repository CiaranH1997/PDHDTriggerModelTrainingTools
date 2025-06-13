import uproot
import awkward as ak
import numpy as np
import pandas as pd
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

def read_tp_data(filename):
    # Open the ROOT file.
    file = uproot.open(filename)
    
    # Define a mapping from APA labels to tree names.
    apa_trees = {
        "APA1": "SmallTriggerTPCInfoDisplay/TPWindowAPA1Tree",
        "APA2": "SmallTriggerTPCInfoDisplay/TPWindowAPA2Tree",
        "APA3": "SmallTriggerTPCInfoDisplay/TPWindowAPA3Tree",
        "APA4": "SmallTriggerTPCInfoDisplay/TPWindowAPA4Tree"
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
            for win_idx, (win_time, win_channel, win_charge, windows_tot, windows_adcpeak) in enumerate(
                zip(windows_times, windows_channels, windows_charges, windows_tot, windows_adcpeak)
            ):
                tp_list = []
                # Each win_time, win_channel, win_charge is a list of integers.
                for t, ch, cq, tt, cp in zip(win_time, win_channel, win_charge, windows_tot, windows_adcpeak):
                    tp_list.append({"Time_peak": t, "ChannelID": ch, "ADC_integral": cq, "ToT": tt, "ADC_peak": cp})
                results[event_id][apa][win_idx] = tp_list
    return results

def read_neutrino_tp_data(filename):
    # Open the ROOT file.
    file = uproot.open(filename)
    
    # --- Step 1. Build a mapping from event id to neutrino APA using GenieTruth ---
    gt_tree = file["SmallTriggerTPCInfoDisplay/GenieTruth"]
    gt_event_ids = gt_tree["EventIterator"].array(library="ak")
    gt_APAs = gt_tree["APA"].array(library="ak")
    
    # Create a mapping: event_id -> APA (integer)
    nu_truth = {}
    for eventid, apa in zip(ak.to_list(gt_event_ids), ak.to_list(gt_APAs)):
        nu_truth[int(eventid)] = int(apa)
    
    tree_name = "SmallTriggerTPCInfoDisplay/TPNuWindowTree"
    
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
        event_id = int(evt) - 1  # Ensure event_id is non-negative and avoid unnecessary adjustments
                
        #print(f'APA = {apa[event_id]}')
        if len(windows_times) == 0:
            current_apa = apa[i] if i < len(apa) else "Unknown"
            print(f'Window has no TPs! Event ID: {event_id}, APA: {current_apa}')
            continue  # No window data for this event.
              
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

    return results

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
                #channel_data = np.array([tp["ChannelID"] for tp in tp_list])
                time_data = np.array([tp["Time_peak"] for tp in tp_list])
                #adc_data = np.array([tp["ADC_integral"] for tp in tp_list])
        
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

                for row in df.itertuples():
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
                    hist_2d /= df.groupby(["Time_bin", "Channel_bin"]).size().values.reshape(hist_2d.shape)
                
                binned_data[event_id][apa].append(hist_2d)

    return binned_data