from brian2.units import *

N_realizations = 12
N_trials = 9

# reduced amount for calculations over different R_ee
N_realizations_Ree = 1
N_trials_Ree = 9

N_exc = 4000
N_inh = 1000 
N_cluster = 80

dt = 0.1*ms
duration = 3*second
after_duration=1.5*second

V = {
    'v_threshold': 1.,
    'v_reset': 0.
}

taus = {
    'refractory_period': 5*ms,
    'tau_exc': 15*ms,
    'tau_inh': 10*ms,
    'tau_2_exc': 3*ms,
    'tau_2_inh': 2*ms,
    'tau_1': 1*ms
    }

mus = {
    'mu_exc_low': 1.1,
    'mu_exc_high': 1.2,
    'mu_inh_low': 1.,
    'mu_inh_high': 1.05
    }

J_uniform = {
    'J_EE': 0.024,
    'J_EI': -0.045,
    'J_IE': 0.014,
    'J_II': -0.057,
    }

J_cluster = {
    'J_EE': 0.024,
    'J_EI': -0.045,
    'J_IE': 0.014,
    'J_II': -0.057,
    'scale': 1.9
    }

P_uniform = {
    'p_EE': 0.2,
    'P_EI': 0.5,
    'P_IE': 0.5,
    'P_II': 0.5
    }

P_cluster = {
    'p_EE': [0.485, 0.194],    #[p_in, p_out]
    'P_EI': 0.5,
    'P_IE': 0.5,
    'P_II': 0.5
    }

equation = """dv/dt = (mu - v)/tau_m + F: 1 (unless refractory)
dx/dt = -x/tau_2 : 1
dF/dt = ((x/tau_2)-F)/tau_1 : Hz
mu : 1
tau_m : second
tau_1 : second
tau_2 : second"""
