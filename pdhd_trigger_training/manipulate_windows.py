import numpy as np

# Function to filter out TPs with certain properties (low TOT/SADC, implying noise)
# This function removes individual TPs from windows based on a cut-off value for a specified variable.
# It returns a new dictionary with the filtered TPs.
def filter_tps_in_window(tp_data, var="ADC_integral", cut=0):
    if var not in ("ADC_integral", "ToT", "ADC_peak"):
        raise ValueError("Var must be ADC_integral, ToT or ADC_peak.")
    
    print(f'Chosen variable for filtering {var}')
    
    filtered_results = {}
    
    for event_id in tp_data:
        
        filtered_results[event_id] = {}
        
        for apa in tp_data[event_id]:
            
            if apa not in filtered_results[event_id]:
                filtered_results[event_id][apa] = []
            
            for i, window in enumerate(tp_data[event_id][apa]):
                
                while len(filtered_results[event_id][apa]) <= i:
                    filtered_results[event_id][apa].append([])
                
                tp_list = tp_data[event_id][apa][i]
                filtered_tp_list = []
                for tp in tp_list:
                    var_element = tp[var]
                    if (var_element > cut):
                        filtered_tp_list.append(tp)
                    
                filtered_results[event_id][apa][i] = filtered_tp_list
    
    return filtered_results

# Function to filter out TPs based on the sum of ADC_integral in each window
def filter_single_variable(tp_data, var="ADC_integral", cut=0):
    
    filtered_results = {}
    count_input = 0
    count_output = 0
    for event_id in tp_data:
        
        filtered_results[event_id] = {}
        
        for apa in tp_data[event_id]:
            
            if apa not in filtered_results[event_id]:
                filtered_results[event_id][apa] = []
            
            for i, window in enumerate(tp_data[event_id][apa]):
                count_input += 1
                tp_list = tp_data[event_id][apa][i]
                window_sum_sadc = average_or_sum_single_window(tp_list, var, False)
                
                if window_sum_sadc > cut:
                    count_output += 1
                    filtered_results[event_id][apa].append(tp_list)
    
    return filtered_results, count_input, count_output

# Function to filter out TPs based on the ratio of mean peak to mean ToT in each window
def filter_mean_peak_tot_ratio(tp_data, cut=0):
    
    filtered_results = {}
    count_input = 0
    count_output = 0
    for event_id in tp_data:
        
        filtered_results[event_id] = {}
        
        for apa in tp_data[event_id]:
            
            if apa not in filtered_results[event_id]:
                filtered_results[event_id][apa] = []
            
            for i, window in enumerate(tp_data[event_id][apa]):
                count_input += 1
                tp_list = tp_data[event_id][apa][i]
                window_mean_tot = average_or_sum_single_window(tp_list, "ToT", True)
                window_mean_peak = average_or_sum_single_window(tp_list, "ADC_peak", True)
                
                ratio = window_mean_peak / window_mean_tot
                if ratio > cut:
                    count_output += 1
                    filtered_results[event_id][apa].append(tp_list)
    
    return filtered_results, count_input, count_output

# Function to filter out windows based on the number of TPs in each window
def filter_number_tps(tp_data, cut=0):
    
    filtered_results = {}
    count_input = 0
    count_output = 0
    for event_id in tp_data:
        
        filtered_results[event_id] = {}
        
        for apa in tp_data[event_id]:
            
            if apa not in filtered_results[event_id]:
                filtered_results[event_id][apa] = []
            
            for i, window in enumerate(tp_data[event_id][apa]):
                count_input += 1
                tp_list = tp_data[event_id][apa][i]
                var_data = np.array([tp["ADC_integral"] for tp in tp_list])
                window_ntps = var_data.size
                
                if window_ntps > cut:
                    count_output += 1
                    filtered_results[event_id][apa].append(tp_list)
    
    return filtered_results, count_input, count_output

# Function to get average ToT, SADC and ADC Peak per window for dict of windows
# Change mean to False to get the sum instead of the average.
def average_or_sum_window(tp_data, var="ADC_integral", mean=True):
    
    if var not in ("ADC_integral", "ToT", "ADC_peak"):
        raise ValueError("Var must be ADC_integral, ToT or ADC_peak.")
    
    print(f'Chosen variable {var}')
    
    ret_array = []
    
    for event_id in tp_data:
        for apa in tp_data[event_id]:
            for i, window in enumerate(tp_data[event_id][apa]):
                tp_list = tp_data[event_id][apa][i]
            
                var_data = np.array([tp[var] for tp in tp_list])
                if mean:
                    var_mean = np.mean(var_data)
                    ret_array.append(var_mean)
                else:
                    var_sum = np.sum(var_data)
                    ret_array.append(var_sum)
                
    return np.array(ret_array)

# Function to get the ratio of the mean of two variables per window
# This is useful for comparing ADC_peak to ToT or ADC_integral
# to ToT, for example.
# It returns an array of ratios for each window.
def average_ratio_window(tp_data, var1="ADC_peak", var2="ToT"):
    
    if var1 not in ("ADC_integral", "ToT", "ADC_peak"):
        raise ValueError("Var must be ADC_integral, ToT or ADC_peak.")
    if var2 not in ("ADC_integral", "ToT", "ADC_peak"):
        raise ValueError("Var must be ADC_integral, ToT or ADC_peak.")
    
    print(f'Chosen variables {var1} and {var2}')
    
    ret_array = []
    
    for event_id in tp_data:
        for apa in tp_data[event_id]:
            for i, window in enumerate(tp_data[event_id][apa]):
                tp_list = tp_data[event_id][apa][i]
            
                var1_data = np.array([tp[var1] for tp in tp_list])
                var2_data = np.array([tp[var2] for tp in tp_list])
                
                var1_mean = np.mean(var1_data)
                var2_mean = np.mean(var2_data)
                ratio = var1_mean / var2_mean
                ret_array.append(ratio)
                
    return np.array(ret_array)

# Function to calculate the average of a single variable in a list of TPs from a single window.
# Change mean to False to get the sum instead of the average.
def average_or_sum_single_window(tp_list, var="ADC_integral", mean=True):
    
    if var not in ("ADC_integral", "ToT", "ADC_peak"):
        raise ValueError("Var must be ADC_integral, ToT or ADC_peak.")
    
            
    var_data = np.array([tp[var] for tp in tp_list])
    if mean:
        var_mean = np.mean(var_data)
        return var_mean
    else:
        var_sum = np.sum(var_data)
        return var_sum

# Function to count the number of TPs in each window
def TP_count(tp_data):
    ret_array = []
    for event_id in tp_data:
        for apa in tp_data[event_id]:
            for i, window in enumerate(tp_data[event_id][apa]):
                tp_list = tp_data[event_id][apa][i]
                var_data = np.array([tp["ADC_integral"] for tp in tp_list])
                ret_array.append(var_data.size)
                
    return np.array(ret_array)

# Function to determin if a window should be cut based on a variable value
# This is useful for filtering out windows with high ADC_integral, for example.
# It returns True if the variable value is greater than the cut-off value.
def make_cut(window, var=0, var_cut=100000):
    if window[var] > var_cut:
        return True
    else:
        return False

# Function to filter out windows based on a variable value
# This is useful for filtering out windows with high ADC_integral, for example.
# It returns the filtered X_val and y_val arrays.
# X_val should be a stacked array of windows, and y_val should be the corresponding labels.
# The var parameter specifies the variable to filter on, and var_cut is the cut-off value.
# The function returns the filtered X_val and y_val arrays.
def var_filter(X_val, y_val, var=0, var_cut=100000):
    
    mask = np.array([make_cut(window, var, var_cut) for window in X_val])
    filtered_X_val = X_val[mask]
    filtered_y_val = y_val[mask]
    
    return filtered_X_val, filtered_y_val