import setuptools
import numpy as np
from brian2 import *
prefs.codegen.target = "numpy"

class CorticalNetwork():
    
    def __init__(self, equations, N_exc, N_inh, V, taus, mus, synaptic_strengths, probabilities, N_cluster=None, is_cluster=True, method='euler'):
        """
        Initialises network parameters.
        :param equations: differential equations defining the network
        :param N_exc: number of excitatory neurons
        :param N_inh: number of inhibitory neurons
        :param V: dictionary of threshold voltage, reset voltage
        :param taus: dictionary of refractory period, tau_m, tau_1, tau_2 for both exc. and inh. neuron types
        :param mus: dictionary of mu intervals for both exc. and inh. neuron types
        :param synaptic_strengths: dictionary of synaptic strengths according to network type
        :param probabilities: dictionary of probabilities of forming a connection according to network type
        :param N_cluster: size of a cluster
        :param is_cluster: Boolean to define if the network is clustered or not
        :param method: numerical integration method
        """
        self.equations = equations
        self.N_exc = N_exc
        self.N_inh = N_inh
        self.N_cluster = N_cluster

        self.v_threshold = V['v_threshold']
        self.v_reset = V['v_reset']

        self.refractory_period = taus['refractory_period']
        self.tau_exc = taus['tau_exc']
        self.tau_inh = taus['tau_inh']
        self.tau_2_exc = taus['tau_2_exc']
        self.tau_2_inh = taus['tau_2_inh']
        self.tau_1 = taus['tau_1']

        self.mu_exc_low = mus['mu_exc_low']
        self.mu_exc_high = mus['mu_exc_high']
        self.mu_inh_low = mus['mu_inh_low']
        self.mu_inh_high = mus['mu_inh_high']

        self.synaptic_strengths = synaptic_strengths
        self.probabilities = probabilities

        self.is_cluster = is_cluster
        self.method = method

    def initialise_neuron_group(self, n_neurons, mu_high, mu_low, tau_m, tau_2, v_threshold=None, v_reset=None, refractory_period=None, tau_1=None, dt='0.1*ms', voltage='rand()'):
        """
        Initialises a group of neurons.
        :param n_neurons: number of neurons in the group
        :param mu_high: Higher limit of mu value
        :param mu_low: Lower limit of mu value
        :param tau_m: tau_m value
        :param tau_2: tau_2 value
        :param v_threshold: voltage threshold value
        :param v_reset: voltage reset value
        :param refractory_period: length of the refractory period
        :param tau_1: tau_1 value
        :param dt: the time step
        :param voltage: initial voltage value
        :return: initialised neuron group
        """

        if v_threshold is None: v_threshold=self.v_threshold
        if v_reset is None: v_reset=self.v_reset
        if refractory_period is None: refractory_period=self.refractory_period
        if tau_1 is None: tau_1=self.tau_1

        neuron_group = NeuronGroup(n_neurons, self.equations, threshold='v>%f'%v_threshold, reset='v=%f'%v_reset, refractory=refractory_period, method=self.method)
        neuron_group.mu = np.random.uniform(mu_low, mu_high, size=n_neurons)
        neuron_group.tau_m = tau_m
        neuron_group.tau_1 = tau_1
        neuron_group.tau_2 = tau_2
        neuron_group.v = voltage

        return neuron_group

    def connect(self, source, destination, synaptic_strength, probability, condition='i!=j'):
        """
        Creates a synaptic connection between neuron groups.
        :param source: source neuron group
        :param destination: destination neuron group
        :param synaptic_strength: synaptic strength to update after every pre-synaptic spike
        :param probability: probability of forming a connection
        :param condition: string of any conditions for forming a connection
        :return: defined synapse brian2 object
        """
        
        on_pre_ = 'x_post += %f'%synaptic_strength
        synapse = Synapses(source, destination, on_pre=on_pre_)
        synapse.connect(condition=condition, p = probability)

        return synapse

    def build_network(self, neuron_group_a, neuron_group_b, synaptic_strengths, probabilities, is_cluster, N_cluster=None, scale_synaptic_strength=1):
        """
        Forms connections between neuron groups based on if or not a cluster
        :param neuron_group_a: a brian2 NeuronGroup object
        :param neuron_group_b: a brian2 NeuronGroup object
        :param synaptic_strengths: dictionary of synaptic strengths according to network type
        :param probabilities: dictionary of probabilities of forming a connection according to network type
        :param is_cluster: Boolean to define if the network is clustered or not
        :param N_cluster: size of a cluster
        :param scale_synaptic_strength: scaling synaptic strength multiplier factor for clustered network 
        :return: all connections formed as synapse brian2 objects
        """
        a_in = a_a = a_b = b_a = b_b = None
        ss = list(synaptic_strengths.values())
        p = list(probabilities.values())

        if(is_cluster):
            # intra cluster
            intra_cluster_condition = 'i!=j and (floor(i/%d)==floor(j/%d))'%(N_cluster, N_cluster)
            a_in = self.connect(neuron_group_a, neuron_group_a, ss[0]*scale_synaptic_strength, p[0][0], intra_cluster_condition)
            # inter cluster
            inter_cluster_condition = 'i!=j and not (floor(i/%d)==floor(j/%d))'%(N_cluster, N_cluster)
            a_a = self.connect(neuron_group_a, neuron_group_a, ss[0], p[0][1], inter_cluster_condition)
        else:
            a_a = self.connect(neuron_group_a, neuron_group_a, ss[0], p[0])
        
        a_b = self.connect(neuron_group_b, neuron_group_a, ss[1], p[1])
        b_a = self.connect(neuron_group_a, neuron_group_b, ss[2], p[2])
        b_b = self.connect(neuron_group_b, neuron_group_b, ss[3], p[3])

        return a_in, a_a, a_b, b_a, b_b

    def run_network(self, duration_1, duration_2=0, N_realizations=1, N_trials=1,  N_split=0, monitor=True):
        """
        Runs a simulation of the network while monitoring(optional) it.
        :param duration_1: duration of the simulation
        :param duration_2: duration of the simulation post monitoring begins
        :param N_realizations: number of realizations of simulation runs
        :param N_trials: number of trials per realization
        :param N_split: number of neurons to monitor
        :param monitor: boolean to choose if variables are to be monitored during simulation
        :return: brian2 monitored objects and spike trains over all realizations
        """
        spike_train_realization = []
        state_monitor_excitatory = None
        spike_monitor_excitatory = None

        for realization in range(N_realizations):
            start_scope()

            excitatory = self.initialise_neuron_group(n_neurons=self.N_exc, mu_high=self.mu_exc_high, mu_low=self.mu_exc_low, tau_m=self.tau_exc, tau_2=self.tau_2_exc)
            inhibitatory = self.initialise_neuron_group(n_neurons=self.N_inh, mu_high=self.mu_inh_high, mu_low=self.mu_inh_low, tau_m=self.tau_inh, tau_2=self.tau_2_inh)
            if self.is_cluster:
                a, b, c, d, e =self.build_network(excitatory, inhibitatory, self.synaptic_strengths, self.probabilities, self.is_cluster, self.N_cluster, self.synaptic_strengths['scale'])
            else:
                _, a, b, c, d = self.build_network(excitatory, inhibitatory, self.synaptic_strengths, self.probabilities, self.is_cluster)

            net = Network(collect())
            net.run(duration_1)

            if monitor:
                excitatory_split = excitatory[:N_split]
                state_monitor_excitatory = StateMonitor(excitatory, 'v', record=True)
                spike_monitor_excitatory = SpikeMonitor(excitatory_split)
                net.add([state_monitor_excitatory, spike_monitor_excitatory])
            
            net.store()

            spike_train_trials = []
            for trial in range(N_trials):
                net.restore()
                net.run(duration_2)
                spike_train_trials.append(spike_monitor_excitatory.spike_trains())

            spike_train_realization.append(spike_train_trials)

        return state_monitor_excitatory, spike_monitor_excitatory, spike_train_realization