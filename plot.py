import matplotlib.pyplot as plt

def voltage_trace_plot(state_monitor, neuron_type='excitatory', network_type=''):
    """
    Plots the voltage trace of a Neuron.
    :param state_monitor: Brian2 state monitor object of a simulated neuron group
    :param network: type of network (uniform/clustered)
    """
    plt.figure(figsize=(12,5))
    plt.plot(state_monitor.t, state_monitor.v[0])
    plt.xticks([1.5, 2.5, 3.5],['0', '1', '2'])
    plt.yticks([0,1],['-65','-50'])
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.title('Voltage trace of an %s neuron in a %s Network'%(neuron_type, network_type))
    plt.show()

def spike_raster_plot(spike_monitor, neuron_type='excitatory', network_type=''):
    """
    Plots the spike raster of a neuron group.
    :param state_monitor: Brian2 spike monitor object of a simulated neuron group
    :param network: type of network (uniform/clustered)
    """
    plt.figure(figsize=(10,6))
    plt.plot(spike_monitor.t, spike_monitor.i, '.k', markersize=1)
    plt.xticks([1.5, 2.5, 3.5],['0', '1', '2'])
    plt.yticks([])
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron')
    plt.title('Spike raster of %s neurons in a %s Network'%(neuron_type, network_type))
    plt.show()