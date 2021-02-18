from brian2 import *
import numpy as np
import pandas as pd

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

	number_windows = int(duration_analysis/window_size)
	windows = np.linspace(after_duration/second,duration/second,number_windows)

	fano_count = np.zeros((N_realizations,N_trials,N_exc, number_windows))


	for realization in range(N_realizations):
		for trial in range(N_trials):
			for neuron in spike_train_realization[realization][trial]:
				fano_windows = []
				for window in windows:
					temp_count = np.sum(np.logical_and(spike_train_realization[realization][trial][neuron]/second > window,spike_train_realization[realization][trial][neuron]/second < (window +window_size)))
					
					fano_windows.append(temp_count)
					
						
				fano_count[realization][trial][neuron] = np.asarray(fano_windows)
				
				
	np.seterr(divide='ignore',invalid='ignore')			
	fano_factor = np.var(fano_count,axis=(1,3))/(np.mean(fano_count,axis=(1,3)))
		
	
#	mean_fano = np.mean(fano_count,axis=1)
	fano_flat = fano_factor.flatten()

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
		fano_over_windows.append(np.nanmean(temp_fano))
		
	return diff_windows, fano_over_windows

def get_spike_train_windowed(spike_train, after_duration, duration, N_trials, N_realizations, N_exc, window_duration=0.05):
    """
    Modified the spike train and stores it over a fixed window duration
    :param spike_train: Brian2 spike train object in a list for all trials, all realizations and all neurons
    :param after_duration: count begins after this duration in seconds
	:param duration: duration of the whole simulation in seconds
	:param N_realizations: number of realizations of simulation runs
	:param N_trials: number of trials per realization
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


def get_autocorrelation(windowed_spike_train,N_realizations,N_trials, N_exc):
	"""
	Get average autocorrelation for a windowed spike train with lags between -200 and 200 ms (-100 * window size = 2ms) 
	:param windowed_spike_train: Windowed spike train in 2ms windows
	:param N_realizations: number of realizations
	:param N_trials: number os trials per realizations
	:param N_exc: number of excitatory neurons in the network
	:return: autocorrelation 
	"""
	autocorrelation = []
	
	for realization in range(N_realizations):
		for trial in range(N_trials):
			for neuron in range(N_exc): 
				neuron_autocorrelation = []
				for lag in range(-100,100):            
					s = pd.Series(windowed_spike_train[realization][trial][neuron])
					neuron_autocorrelation.append(s.autocorr(lag = lag))
            
				autocorrelation.append(neuron_autocorrelation)
				

	autocorrelation = np.asarray(autocorrelation,dtype=double)
	acorr = np.nanmean(autocorrelation,axis=0)
	
	return acorr
<<<<<<< HEAD
	
def get_crosscorrelation(windowed_spike_train, N_realizations, N_trials, N_exc, N_cluster, network_type='', save = True):
	'''
	Get crosscorrelation functions between neuron pairs belonging to the same cluster for a windowed spike train and saves it in a file named "neurons_crosscor_%f %network type"
	:param windowed_spike_train: Windowed spike train in 2ms windows
	:param N_realizations: number of realizations
	:param N_trials: number os trials per realizations
	:param N_exc: number of excitatory neurons in the network
	:param N_cluster: number of neurons inside one cluster
	:param network_type: type of network (uniform/clustered) 
	:return: crosscorrelation arrays
	'''
	
	n_clusters = int(N_exc/N_cluster)
	if save:
		file_name = './data/neurons_crosscor_%s.pkl'%network_type	
		temp = open(file_name, 'wb')
		
	for realization in range(N_realizations):
		for trial in range(N_trials):
		    neurons_crosscor = []
		    for cluster in range(n_clusters):
		        for neuron_a in range((int(cluster*cluster_size)),int((cluster+1)*cluster_size)):   
		            for neuron_b in range((int(cluster*cluster_size)),int((cluster+1)*cluster_size)):
		                
		                if neuron_a < neuron_b:            
		                    a = np.asarray(windowed_uniform[realization][trial][neuron_a])
		                    b = np.asarray(windowed_uniform[realization][trial][neuron_b])
		                    neurons_crosscor.append(np.correlate(a,b,"full")) 
		                    
		    if save:
				temp = open(file_name, 'ab')
				pickle.dump(neurons_crosscor, temp) 


	return neurons_crosscor
	
=======

def get_correlation(spike_train, N_realizations, N_trials, N_exc):
	"""
	Calculates the coorelation between all pairs of excitatory neurons in a spike train
	:param spike_train: spike train of neurons
	:param N_realizations: number of realizations
	:param N_trials: number os trials per realizations
	:param N_exc: number of excitatory neurons in the network
	:return: correlation array for all pairs of excitatory neurons over all trials and realisations
	"""
    correlation_coeff = np.ndarray((N_realizations,N_trials,N_exc, N_exc))
    for nr in range(N_realizations):
        for nt in range(N_trials):
            for i, _ in enumerate(spike_train[nr][nt]):
                for j, _ in enumerate(spike_train[nr][nt]):
                    if(j<=i):
                        corr_temp = np.corrcoef(spike_train[nr][nt][i], spike_train[nr][nt][j], rowvar=False)[0][1]
                        correlation_coeff[nr][nt][i][j] = corr_temp
                        correlation_coeff[nr][nt][j][j] = corr_temp
						
    return correlation_coeff

>>>>>>> c5939c5af1bb79c97f4b15cf394fce13f4256d2b


