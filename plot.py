from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

def voltage_trace_plot(state_monitor, after_duration, neuron_type='excitatory', network_type='', neuron_index=10):
    """
    Plots the voltage trace of a Neuron.
    :param state_monitor: Brian2 state monitor object of a simulated neuron group
    :param after_duration: plot begins after this duration
    :param network: type of network (uniform/clustered)
    """
    index = np.where(state_monitor.t/second > after_duration)[0]

    plt.figure(figsize=(12,5))
    plt.plot(state_monitor.t[index], state_monitor.v[neuron_index][index])
    plt.xticks(list(xticks()[0]), list(xticks()[0]-after_duration))
    plt.yticks([0,1],['-65','-50'])
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.title('Voltage trace of an %s neuron in a %s Network'%(neuron_type, network_type))
    plt.show()

def spike_raster_plot(spike_monitor, after_duration, neuron_split=1600, neuron_type='excitatory', network_type='', grey = 0, stim_begin=0, stim_end = 0):
    """
    Plots the spike raster of a neuron group.
    :param state_monitor: Brian2 spike monitor object of a simulated neuron group
    :param after_duration: plot begins after this duration
    :param network: type of network (uniform/clustered)
    :param grey: if not zero, colors the bottom grey Neurons with grey background
    """

    fig, (ax0, ax1) = plt.subplots(2,1,figsize=(10,6), gridspec_kw={'height_ratios':[5,1]})
    index = np.logical_and(spike_monitor.t > after_duration, spike_monitor.i<neuron_split)
    ax0.plot(spike_monitor.t[index], spike_monitor.i[index], '.k', markersize=1)
    ax0.set_yticks([])
    ax0.set_xticks([])
    ax0.set_ylabel('Neuron')
    ax0.set_title('Spike raster of %s neurons in a %s Network'%(neuron_type, network_type))
    if grey != 0:
        ax0.axhspan(0, grey, facecolor='0.2', alpha=0.1)
        end_time = np.max(spike_monitor.t[index])
        begin_time = np.min(spike_monitor.t[index])
        x = np.linspace(begin_time, end_time, 1000)
        y = np.zeros(len(x))
        y[np.logical_and(x>(stim_begin+after_duration) , x < (stim_end+after_duration))] = 1
        ax1.plot(x-after_duration,y)
#        ax0.set_xticks(list(xticks()[0]), list(xticks()[0]-after_duration))
        ax1.set_ylabel("Stim")
        ax1.set_yticks([])
    plt.xlabel('Time (s)')
    plt.show()
    
def firing_rate_histogram_plot(flat_rates_a,flat_rates_b, color_a='grey',color_b='green', bin_size_a=70, bin_size_b=280):
    """
    Plots the histogram of firing rates.
    :param flat_rates_histogram: flattened array of counts of firing rates averaged over trials for all realizations for all neurons
    :param network: type of network (uniform/clustered)
    :param color: color for the histogram
    """
    plt.hist(x=flat_rates_a, bins=bin_size_a, histtype='step', color=color_a)
    plt.hist(x=flat_rates_b, bins=bin_size_b, histtype='step', color=color_b)

    plt.xlabel('Rate [Hz]',fontsize = 15)
    plt.ylabel('Count',fontsize = 15)

    plt.title("Histogram of Firing Rates",fontsize = 15)
    plt.plot(np.mean(flat_rates_a), plt.ylim()[1]/2, 'v', color = color_a, label='mean uniform')
    plt.plot(np.mean(flat_rates_b), plt.ylim()[1]/2, 'v', color = color_b, label='mean cluster')

    plt.xlim(0,10)
    plt.legend()
    plt.show()

def fano_factor_histogram_plot(fano_flat_a,fano_flat_b, bins_a = 10, bins_b = 10):
    """
    Plots the histogram of fano factors.
    :param fano_flat: flattened array of counts of fano_factos averaged over trials and 100ms windows for all realizations for all neurons
    :param network: type of network (uniform/clustered)
    :param color: color for the histogram
    """
    plt.hist(x=fano_flat_a, bins=bins_a, histtype='step', color='grey', label = 'uniform')
    plt.hist(x=fano_flat_b, bins=bins_b, histtype='step', color='green', label = 'cluster')
    plt.ylabel('Count',fontsize = 15)
    plt.xlabel('Fano Factors',fontsize = 15)
    plt.xlim(0,3)
    plt.title("Histogram of Fano Factos",fontsize = 15)
    plt.legend()
    plt.show()  


def fano_factor_windows_plot(window, fano_over_windows_a, fano_over_windows_b):
    """
    Plots fano factors for different window sizes.
    :param diff_windows: different window sizes used 
    :param fano_var_size: fano factors for different window sizes
    :param network: type of network (uniform/clustered)
    :param color: color for the histogram
    """


    plt.plot(window,fano_over_windows_a, color='grey', label = 'uniform')
    plt.plot(window,fano_over_windows_b, color='green', label = 'cluster')
    plt.ylim(0,2.5)
    plt.xlim
    plt.ylabel('Fano Factor',fontsize = 15)
    plt.xlabel('Window Size [sec]',fontsize = 15)
    plt.title("Fano Factors for Different Window sizes",fontsize = 15)
    plt.legend()
    plt.show()  

def fano_factor_over_ree_plot(fano_ree_avg, r_ee):
    #print(fano_Ree)
#    print(fano_Ree_avg)
    plt.scatter(r_ee,fano_ree_avg)
    plt.xlabel("Ree", fontsize = 15)
    plt.ylabel("fano-factor", fontsize = 15)
    plt.ylim(0,7)
    plt.title("Fano Factor for different R_ee", fontsize = 15)
#    plt.legend()
    plt.show()

def fano_factor_over_time_plot(fano_over_time_all, fano_over_time_stim, fano_over_time_no_stim):
    fig, axs = plt.subplots(1)
    x = np.linspace(0,2.4, len(fano_over_time_all))
    axs.plot(x, fano_over_time_all)
    axs.set_title("Fano Factor over time for all neurons")
    axs.set_xlabel("time in seconds")
    axs.set_ylabel("fano factor")
    
    fig, axs = plt.subplots(1,2, figsize = (12,5))
    x = np.linspace(0,2.4, len(fano_over_time_all))
    axs[0].plot(x, fano_over_time_stim)
    axs[0].set_title("Fano Factor over time for stimulated neurons")
    axs[0].set_xlabel("time in seconds")
    axs[0].set_ylabel("fano factor")
    axs[1].plot(x, fano_over_time_no_stim)
    axs[1].set_title("Fano Factor over time for non-stimulated neurons")
    axs[1].set_xlabel("time in seconds")
    axs[1].set_ylabel("fano factor")


