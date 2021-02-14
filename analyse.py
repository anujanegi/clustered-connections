from brian2 import *
import numpy as np

def get_flat_firing_rate(spike_train_realization, after_duration, duration, N_trials, N_realizations, N_exc,  neuron_type='excitatory', network_type=''):

	"""
	Calculate the flattened array of counts of firing rates for each neuron averaged over trials in all realizations 
	:param spike_train_realization: Brian2 spike train object in a list for all trials, all realizations and all neurons
	:param after_duration: count begins after this duration
	:param duration: duration of the whole simulation
	:param N_realizations: number of realizations of simulation runs
	:param N_trials: number of trials per realization
	:param N_exc: number of excitatory neurons
	:param network: type of network (uniform/clustered)
	:return: flattened firing rates
	"""

	duration_analysis = (duration - after_duration)/second

	counts = np.zeros((N_realizations,N_trials,N_exc))
	for realization in range(N_realizations):
		for trial in range(N_trials):
			for neuron in spike_train_realization[realization][trial]:
					counts[realization][trial][neuron] = np.sum(spike_train_realization[realization][trial][neuron]/second>after_duration/second)
		
		
		flat_rates_histogram = np.asarray(np.mean(counts,axis=1)/duration_analysis).flatten()
	return flat_rates_histogram
 	       
		   
def get_fano_factor(spike_train_realization, after_duration, duration, N_trials, N_realizations, N_exc,  neuron_type='excitatory', network_type='', window_size = 0.1 ):

	"""
	Calculates the flattened array of counts of fano factors for each neuron averaged over trials and windows of window_size in all realizations 
	:param spike_train_realization: Brian2 spike train object in a list for all trials, all realizations and all neurons
	:param after_duration: count begins after this duration in seconds
	:param duration: duration of the whole simulation in seconds
	:param N_realizations: number of realizations of simulation runs
	:param N_trials: number of trials per realization
	:param N_exc: number of excitatory neurons
	:param network: type of network (uniform/clustered)
	:param window_size: window size used to calculate the fano factors in seconds
	:return: flattened array of fano factors
	"""
	duration_analysis = (duration - after_duration)/second
	fano_count = np.zeros((N_realizations,N_trials,N_exc))

	number_windows = int(duration_analysis/window_size)
	windows = np.linspace(after_duration/second,duration/second,number_windows)

	for realization in range(N_realizations):
		for trial in range(N_trials):
			for neuron in spike_train_realization[realization][trial]:
				fano_windows = []
				for window in windows:
					temp_count = np.sum(np.logical_and(spike_train_realization[realization][trial][neuron]/second > window,spike_train_realization[realization][trial][neuron]/second < (window +window_size)))
					
					fano_windows.append(temp_count)
					
				np.seterr(divide='ignore',invalid='ignore')			
				fano_count[realization][trial][neuron] = np.var(np.asarray(fano_windows))/(np.mean(np.asarray(fano_windows)))
		
		
	mean_fano = np.mean(fano_count,axis=1)
	fano_flat = mean_fano.flatten()

	return fano_flat

def get_fano_factor_windows(spike_train_realization, after_duration, duration, N_trials, N_realizations, N_exc,  neuron_type='excitatory', network_type=''):
	"""
	Calculate the fano factors for window sizes between 0.025 and 0.2 seconds averaged for the entire set of neuron_type and network_type
	:param spike_train_realization: Brian2 spike train object in a list for all trials, all realizations and all neurons
	:param after_duration: count begins after this duration in seconds
	:param duration: duration of the whole simulation in seconds
	:param N_realizations: number of realizations of simulation runs
	:param N_trials: number of trials per realization
	:param N_exc: number of excitatory neurons
	:param network: type of network (uniform/clustered)
	:return: different window sizes and fano factors for all window sizes
	"""
	diff_windows = np.linspace(0.025,0.200,8)
	fano_over_windows = []

	for window_size in diff_windows:
		temp_fano = get_fano_factor(spike_train_realization, after_duration, duration, N_trials, N_realizations, N_exc,  neuron_type='excitatory', network_type='', window_size = window_size )		
		fano_over_windows.append(np.mean(np.nan_to_num(temp_fano)))
		
	return diff_windows, fano_over_windows

def get_spike_train_windowed(spike_train, window_duration=0.05):
    """
    Modified the spike train and stores it over a fixed window duration
    :param spike_train_realization: Brian2 spike train object in a list for all trials, all realizations and all neurons
    :param window_duration: window size
    :return: modified spike train
    """
    N_windows = int((duration/second - after_duration)/window_duration) #1.5s/50ms
    windowed_spike_train = np.zeros((N_realizations,N_trials,N_exc, N_windows))

    for nr in range(N_realizations):
        for nt in range(N_trials):
            for i in spike_train[nr][nt]:
                neuron = spike_train[nr][nt][i]
                for window in range(N_windows):
                    index = np.logical_and(neuron/second > after_duration+(window*window_duration), neuron/second<after_duration+((window+1)*window_duration))
                    windowed_spike_train[nr][nt][i][window] = sum(index)

    return windowed_spike_train








