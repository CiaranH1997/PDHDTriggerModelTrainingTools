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
            
            # Loop over the 10 windows.
            for win_idx, (win_time, win_channel, win_charge) in enumerate(
                zip(windows_times, windows_channels, windows_charges)
            ):
                tp_list = []
                # Each win_time, win_channel, win_charge is a list of integers.
                for t, ch, cq in zip(win_time, win_channel, win_charge):
                    tp_list.append({"Time_peak": t, "ChannelID": ch, "ADC_integral": cq})
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
        
    # Convert awkward arrays to lists.
    event_ids_list    = ak.to_list(event_ids)
    apa_list          = ak.to_list(apa)
    windows_times     = ak.to_list(times)
    windows_channels  = ak.to_list(channels)
    windows_charges   = ak.to_list(charges)
        
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
        
        tp_list = []
        
        for t, ch, cq in zip(event_windows_times[0], event_windows_channels[0], event_windows_charges[0]):
            tp_list.append({"Time_peak": t, "ChannelID": ch, "ADC_integral": cq})
            results[event_id][apa_key][0] = tp_list

    return results

def plot_th2d_y_projections(th2d_list):
    """
    Takes a list of TH2D histograms, projects each onto the Y-axis,
    and plots the resulting TH1D histograms in a TCanvas.
    
    Parameters:
        th2d_list (list of ROOT.TH2D): List of TH2D histograms.
    """
    if not th2d_list:
        print("Error: Empty list of histograms.")
        return
    
    num_hist = len(th2d_list)
    rows = 5
    cols = 2
    
    if num_hist != rows * cols:
        print(f"Warning: Expected {rows * cols} histograms, got {num_hist}. Layout may be incorrect.")
    
    ROOT.gStyle.SetOptStat(0)
    
    canvas = ROOT.TCanvas("canvas", "TH2D Y Projections", 1200, 1500)
    canvas.Divide(cols, rows)
    
    hist_list = []
    
    for i, th2d in enumerate(th2d_list):
        canvas.cd(i + 1)
        hist_proj = th2d.ProjectionY(f"projY_{i}")
        hist_proj.SetTitle(f"ADC with Time {i}")
        hist_proj.SetLineColor(4)
        hist_proj.SetLineWidth(2)
        #hist_proj.GetYaxis().SetLabelSize(0.045)
        hist_proj.Draw("HIST")
        hist_list.append(hist_proj)
    
    canvas.Update()
    canvas.Draw()
    canvas.Print("/eos/user/c/chasnip/DUNE/ProtoDUNE_BSM/TriggerTrainingData/CosmicTPs_1DTimeProjection_Ev.pdf")
    return canvas, hist_list


# Function to bin the TPs in a time series array
def bin_windows_by_time(sub_grouped_data, bin_width=1000, window_length=20000):                
    binned_data = {}

    for event_id in sub_grouped_data:
        binned_data[event_id] = {}

        for apa in sub_grouped_data[event_id]:
            binned_data[event_id][apa] = []  # Store all sub-events as lists
            #for sub_event in sub_grouped_data[event_id][apa]:
            for i, window in enumerate(sub_grouped_data[event_id][apa]):
                
                #time_data = np.array(window["Time_peak"])
                tp_list = sub_grouped_data[event_id][apa][i]
                channel_data = np.array([tp["ChannelID"] for tp in tp_list])
                time_data = np.array([tp["Time_peak"] for tp in tp_list])
                adc_data = np.array([tp["ADC_integral"] for tp in tp_list])
        
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

                # Sum ADC_Integral within each time bin
                binned_adc = df.groupby("Time_bin")["ADC_integral"].sum()

                # Convert to a NumPy array (handling missing bins with fillna)
                binned_array = binned_adc.reindex(range(len(bins) - 1), fill_value=0).to_numpy()

                binned_data[event_id][apa].append(binned_array)  # Store all sub-events

    return binned_data