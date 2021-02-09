from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

def voltage_trace_plot(state_monitor, after_duration, neuron_type='excitatory', network_type=''):
    """
    Plots the voltage trace of a Neuron.
    :param state_monitor: Brian2 state monitor object of a simulated neuron group
    :param after_duration: plot begins after this duration
    :param network: type of network (uniform/clustered)
    """
    index = np.where(state_monitor.t/second > after_duration)[0]

    plt.figure(figsize=(12,5))
    plt.plot(state_monitor.t[index], state_monitor.v[0][index])
    plt.xticks(list(xticks()[0]), list(xticks()[0]-after_duration))
    plt.yticks([0,1],['-65','-50'])
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.title('Voltage trace of an %s neuron in a %s Network'%(neuron_type, network_type))
    plt.show()

def spike_raster_plot(spike_monitor, after_duration, neuron_type='excitatory', network_type=''):
    """
    Plots the spike raster of a neuron group.
    :param state_monitor: Brian2 spike monitor object of a simulated neuron group
    :param after_duration: plot begins after this duration
    :param network: type of network (uniform/clustered)
    """
    index = np.where(spike_monitor.t/second > after_duration)[0]
    plt.figure(figsize=(10,6))
    plt.plot(spike_monitor.t[index], spike_monitor.i[index], '.k', markersize=1)
    plt.xticks(list(xticks()[0]), list(xticks()[0]-after_duration))
    plt.yticks([])
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron')
    plt.title('Spike raster of %s neurons in a %s Network'%(neuron_type, network_type))
    plt.show()