import awkward as ak
import time
import numpy as np
import h5py
import ROOT # type: ignore

# Function to plot TH2Ds for each window of a given event and APA
# This function takes a dictionary of time peaks, an event ID, and an APA,
# and plots the time peaks in a 2D histogram for each sub-event.
# It returns the canvas and a list of the histograms.
def plot_windows(tp_dict, event_id, apa, n_bins_ch=None, n_bins_time=50, outfile="./CosmicTPs_TPTimeChImages_Ev_APA1.pdf"):
    
    if event_id not in tp_dict:
        print(f"Event {event_id} not found in the data.")
        return
    if apa not in tp_dict[event_id]:
        print(f"No data for {apa} in event {event_id}.")
        return

    apa_windows = tp_dict[event_id][apa]
    
    # Set APA channel ranges according to your mapping
    apa_ranges = {
        "APA1": (2080, 2560),
        "APA2": (7200, 7680),
        "APA3": (4160, 4640),
        "APA4": (9280, 9760)
    }
    ch_min, ch_max = apa_ranges[apa]
    if n_bins_ch is None:
        n_bins_ch = ch_max - ch_min + 1

    ROOT.gStyle.SetOptStat(0)
        
    # Create a unique canvas name using the time module to avoid conflicts
    canvas_name = "canvas_" + str(int(time.time()))
    canvas = ROOT.TCanvas(canvas_name, f"TH2D per Sub-event for Event {event_id} - {apa}", 1200, 800)
    canvas.Clear()
    n_windows = len(apa_windows)
    # Divide canvas into a grid; e.g. for 10 sub-events, use a 2x5 layout
    # Adjust grid division depending on the number of sub-events
    cols = 2
    rows = (n_windows + cols - 1) // cols
    canvas.Divide(cols, rows)
    
    # Hold references to the histograms
    hist_list = []
    
    # Loop over each sub-event and create a TH2D
    for i, window in enumerate(apa_windows):
        tp_list = apa_windows[i]
        channel_data = np.array([tp["ChannelID"] for tp in tp_list])
        time_data = np.array([tp["Time_peak"] for tp in tp_list])
        adc_data = np.array([tp["ADC_integral"] for tp in tp_list])
        
        if len(time_data) == 0:
            continue
        
        time_min = time_data.min()
        time_max = time_data.max()
        
        hist_name = f"h_event{event_id}_{apa}_sub{i}"
        
        if ROOT.gDirectory.Get(hist_name):
            ROOT.gDirectory.Delete(hist_name + ";1")
            
        hist_title = f"Event {event_id} - {apa} - Sub-event {i}; ChannelID; Time_peak"
        h2d = ROOT.TH2D(hist_name, hist_title, n_bins_ch, ch_min, ch_max, n_bins_time, time_min, time_max)
        h2d.SetDirectory(0)  # Prevent ROOT from keeping track of this histogram
        
        # Fill the histogram using ADC_Integral as weight
        for ch, t, adc in zip(channel_data, time_data, adc_data):
            h2d.Fill(ch, t, adc)
        
        hist_list.append(h2d)
        canvas.cd(i+1)
        h2d.Draw("COLZ")
    
    canvas.Update()
    canvas.Draw()
    canvas.Print(outfile)
    
    return canvas, hist_list  # Return references if further inspection is needed



def plot_windows_from_hdf5(hdf5_file, event_id, apa, n_bins_ch=None, n_bins_time=50, outfile=None):
    """
    Plot TH2D histograms for each window of a given event and APA from HDF5 file.
    """
    # Open file (read-only)
    with h5py.File(hdf5_file, "r") as h5f:
        # Check event exists
        if str(event_id) not in h5f:
            print(f"[❌] Event {event_id} not found in {hdf5_file}")
            return
        event_group = h5f[str(event_id)]

        # Check APA exists for this event
        if apa not in event_group:
            print(f"[❌] No data for {apa} in event {event_id}")
            return
        apa_group = event_group[apa]

        # --- Determine number of windows ---
        # We stored window datasets like "0_time", "1_time", ...
        window_indices = sorted(
            {int(k.split("_")[0]) for k in apa_group.keys() if k.endswith("_time")}
        )

        if len(window_indices) == 0:
            print(f"[⚠️] No windows found for Event {event_id}, {apa}")
            return

        # Set APA channel ranges according to mapping
        apa_ranges = {
            "APA1": (2080, 2560),
            "APA2": (7200, 7680),
            "APA3": (4160, 4640),
            "APA4": (9280, 9760)
        }
        ch_min, ch_max = apa_ranges[apa]
        if n_bins_ch is None:
            n_bins_ch = ch_max - ch_min + 1

        # --- Prepare ROOT canvas ---
        ROOT.gStyle.SetOptStat(0)
        canvas_name = f"canvas_{int(time.time())}"
        canvas = ROOT.TCanvas(canvas_name, f"Event {event_id} - {apa}", 1200, 800)
        n_windows = len(window_indices)
        cols = 2
        rows = (n_windows + cols - 1) // cols
        canvas.Divide(cols, rows)

        hist_list = []

        # --- Loop through each window ---
        for i, widx in enumerate(window_indices):
            try:
                time_data = np.array(apa_group[f"{widx}_time"][:])
                ch_data   = np.array(apa_group[f"{widx}_channel"][:])
                adc_data  = np.array(apa_group[f"{widx}_charge"][:]) \
                    if f"{widx}_charge" in apa_group else np.ones_like(time_data)
            except KeyError as e:
                print(f"[⚠️] Missing dataset for window {widx}: {e}")
                continue

            if len(time_data) == 0:
                continue

            time_min = time_data.min()
            time_max = time_data.max()

            hist_name = f"h_event{event_id}_{apa}_sub{widx}"
            if ROOT.gDirectory.Get(hist_name):
                ROOT.gDirectory.Delete(hist_name + ";1")

            hist_title = f"Event {event_id} - {apa} - Window {widx}; ChannelID; Time_peak"
            h2d = ROOT.TH2D(hist_name, hist_title,
                            n_bins_ch, ch_min, ch_max,
                            n_bins_time, time_min, time_max)
            h2d.SetDirectory(0)

            for ch, t, adc in zip(ch_data, time_data, adc_data):
                h2d.Fill(ch, t, adc)

            hist_list.append(h2d)
            canvas.cd(i+1)
            h2d.Draw("COLZ")

        canvas.Update()
        canvas.Draw()

        if outfile is None:
            outfile = f"./TP_Event{event_id}_{apa}.pdf"
        canvas.Print(outfile)
        print(f"[✅] Plots saved to {outfile}")

    return canvas, hist_list


# Function to plot TH2Ds for each window of a given event and APA
# This function takes a dictionary of time peaks, an event ID, and an APA,
# and plots the time peaks in a 2D histogram for each sub-event.
# It returns the canvas and a list of the histograms.
def plot_nu_windows(tp_dict, event_id, n_bins_ch=None, n_bins_time=50, outfile="./NeutrinoTPs_TPTimeChImages_Ev.pdf"):
    
    event_list = []
    
    #if event_id not in tp_dict:
    while len(event_list) < 10:
        if event_id in tp_dict:
            print(f'event to append = {event_id}')
            event_list.append(event_id)
        event_id += 1

    print(event_list)
    # Set APA channel ranges according to your mapping
    apa_ranges = {
        "APA1": (2080, 2560),
        "APA2": (7200, 7680),
        "APA3": (4160, 4640),
        "APA4": (9280, 9760)
    }

    # Create a unique canvas name using the time module to avoid conflicts
    canvas_name = "canvas_" + str(int(time.time()))
    canvas = ROOT.TCanvas(canvas_name, f"TH2D per Sub-event for Event {event_id}", 1200, 800)
    canvas.Clear()
    n_windows = len(event_list)
    # Divide canvas into a grid; e.g. for 10 sub-events, use a 2x5 layout
    # Adjust grid division depending on the number of sub-events
    cols = 2
    rows = (n_windows + cols - 1) // cols
    canvas.Divide(cols, rows)
    
    # Hold references to the histograms
    hist_list = []
    
    ROOT.gStyle.SetOptStat(0)
    
    # Loop over each sub-event and create a TH2D
    canvas_it = 0
    for i in event_list:
        canvas_it += 1
        APA = "none"
        for apa, (start, end) in apa_ranges.items():
            if apa in tp_dict[i]:
                APA = apa
                break
        print(f'APA = {APA}')
        if APA not in tp_dict[i]:
            print(f"No data for {apa} in event {i}.")
            continue
            
        apa_windows = tp_dict[i][APA]
        # There is only one window per event for neutrino data
        tp_list = apa_windows[0]
        channel_data = np.array([tp["ChannelID"] for tp in tp_list])
        time_data = np.array([tp["Time_peak"] for tp in tp_list])
        adc_data = np.array([tp["ADC_integral"] for tp in tp_list])
        
        time_min = time_data.min()
        time_max = time_data.max()
        
        hist_name = f"h_event{i}_{apa}"
        
        if ROOT.gDirectory.Get(hist_name):
            ROOT.gDirectory.Delete(hist_name + ";1")
            
        # Get bounds for specific APA neutrino occurs in
        ch_min, ch_max = apa_ranges[APA]
        if n_bins_ch is None:
            n_bins_ch = ch_max - ch_min + 1
        
        hist_title = f"Event {i} - {apa}; ChannelID; Time_peak"
        h2d = ROOT.TH2D(hist_name, hist_title, n_bins_ch, ch_min, ch_max, n_bins_time, time_min, time_max)
        h2d.SetDirectory(0)  # Prevent ROOT from keeping track of this histogram
        
        # Fill the histogram using ADC_Integral as weight
        for ch, t, adc in zip(channel_data, time_data, adc_data):
            h2d.Fill(ch, t, adc)
        
        hist_list.append(h2d)
        canvas.cd(canvas_it)
        h2d.Draw("COLZ")
    
    canvas.Update()
    canvas.Draw()
    canvas.Print(outfile)
    
    return canvas, hist_list  # Return references if further inspection is needed

# Function to plot Y projections of a list of TH2D histograms
# This function takes a list of TH2D histograms and plots their Y projections in a grid layout.
# It returns the canvas and a list of the projected histograms.
# The output file is saved as a PDF.
def plot_th2d_y_projections(th2d_list, outfile="./CosmicTPs_1DTimeProjection_Ev.pdf"):
    
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
        hist_proj.Draw("HIST")
        hist_list.append(hist_proj)
    
    canvas.Update()
    canvas.Draw()
    canvas.Print(outfile)
    
    return canvas, hist_list


def plot_nu_windows_from_hdf5(hdf5_file, start_event_id, n_events=10,
                              n_bins_ch=None, n_bins_time=50,
                              outfile="./NeutrinoTPs_TPTimeChImages_Ev.pdf"):
    """
    Plot TH2D histograms for neutrino TP data stored in HDF5.
    It looks for up to `n_events` events starting at `start_event_id`.
    """

    # APA channel ranges
    apa_ranges = {
        "APA1": (2080, 2560),
        "APA2": (7200, 7680),
        "APA3": (4160, 4640),
        "APA4": (9280, 9760)
    }

    # Open HDF5 file
    with h5py.File(hdf5_file, "r") as h5f:
        # --- Collect up to n_events starting at start_event_id ---
        event_list = []
        event_id = start_event_id
        while len(event_list) < n_events:
            if str(event_id) in h5f:
                event_list.append(event_id)
            event_id += 1

        if len(event_list) == 0:
            print(f"[❌] No events found starting at ID {start_event_id}")
            return

        print(f"[ℹ️] Found events to plot: {event_list}")

        ROOT.gStyle.SetOptStat(0)

        # Prepare canvas
        canvas_name = f"canvas_{int(time.time())}"
        canvas = ROOT.TCanvas(canvas_name, f"Neutrino TPs starting {start_event_id}", 1200, 800)
        cols = 2
        rows = (len(event_list) + cols - 1) // cols
        canvas.Divide(cols, rows)

        hist_list = []

        # --- Loop through events ---
        for idx, ev in enumerate(event_list):
            ev_group = h5f[str(ev)]

            # Find which APA this event has (should only be one)
            apa_key = None
            for candidate in apa_ranges.keys():
                if candidate in ev_group:
                    apa_key = candidate
                    break

            if apa_key is None:
                print(f"[⚠️] No APA group found for event {ev}")
                continue

            apa_group = ev_group[apa_key]

            # Read the one window datasets
            try:
                times = np.array(apa_group["0_time"][:])
                channels = np.array(apa_group["0_channel"][:])
                charges = np.array(apa_group["0_charge"][:])
            except KeyError as e:
                print(f"[⚠️] Missing dataset for event {ev}: {e}")
                continue

            if len(times) == 0:
                print(f"[⚠️] Empty TP window for event {ev}")
                continue

            time_min, time_max = times.min(), times.max()
            ch_min, ch_max = apa_ranges[apa_key]
            if n_bins_ch is None:
                nbins_ch_this = ch_max - ch_min + 1
            else:
                nbins_ch_this = n_bins_ch

            hist_name = f"h_event{ev}_{apa_key}"
            if ROOT.gDirectory.Get(hist_name):
                ROOT.gDirectory.Delete(hist_name + ";1")

            hist_title = f"Event {ev} - {apa_key}; ChannelID; Time_peak"
            h2d = ROOT.TH2D(hist_name, hist_title,
                            nbins_ch_this, ch_min, ch_max,
                            n_bins_time, time_min, time_max)
            h2d.SetDirectory(0)

            for ch, t, adc in zip(channels, times, charges):
                h2d.Fill(ch, t, adc)

            hist_list.append(h2d)
            canvas.cd(idx+1)
            h2d.Draw("COLZ")

        canvas.Update()
        canvas.Draw()
        canvas.Print(outfile)
        print(f"[✅] Saved neutrino TP plots to {outfile}")

    return canvas, hist_list
