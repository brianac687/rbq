
import matplotlib.pyplot as plt
import qutip
from qutip import *
import numpy as np
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter1d

# ----------------------------------------------------------------------------------------------------
# alpha_in = input pulse function at time t
# g = coupling term
# k = template
# T = simulation time (max)
# N = number of time steps from 0 to T
def simulate_two_level(alpha_in, g, k, T, N):
    time = np.linspace(0, T, N)
    alpha = alpha_in(time)

    g_vals, k_vals = [], []
    for t in time: 
        g_vals.append(g(t)); k_vals.append(k(t))

    g_vals_smoothed = gaussian_filter1d(g_vals, sigma=10)  # Apply Gaussian smoothing
    def g_smoothed(t):
        if t == time[0]: return 0
        else:            return np.interp(t, time, g_vals_smoothed)  # Interpolate smoothed g values

    ### ---------------------------- HAMILTONIAN ---------------------------------
    H0 = Qobj([[0, 0], [0, 0]]) # time-independent part of Hamiltonian
    H1 = Qobj([[0, 1], [1, 0]])  
    def H1_coeff(t, args): 
        return g_smoothed(t) * alpha_in(t) 
    H = [H0,[H1, H1_coeff]] # Final Hamiltonian (Note: H[1][1](t, None) = H1_coeff at time t)

    # ------------------DISSIPATION OPERATORES (collapse operators)-----------------
    def col_coeff(t, args):  # coefficient function
        return -np.conjugate(g_smoothed(t))
    c_op_list = [[sigmam(), col_coeff]]  # time-dependent collapse term

    ### SIMULATE: starting from the ground state, solve Schrodinger equation
    psi0 = basis(2, 1) # |g> in the sigma z basis = [1, 0]'
    result = mesolve(H, psi0, time, c_op_list) 

    ###--------------------- RESULTS ---------------------------
    prob_ground = [] # probability of finding state in |g>
    prob_excited = [] # probability of finding state in |e>
    S = [] # expectation value of lowering operator
    for i in range(0, len(result.states)):
        p_g, p_e = result.states[i][1][1], result.states[i][0][0] # (with dissipation, so these are density operators) 
        prob_ground = np.append(prob_ground, p_g)
        prob_excited = np.append(prob_excited, p_e)
        S = np.append(S, expect(sigmam(), result.states[i]))

    S_ideal = []
    for i in range(0, len(time)):
        if i == 0: S_ideal.append(0)
        else:      S_ideal.append(np.sqrt(integrate.simps(abs(alpha[0:i])**2, time[0:i])))

    return time, alpha, g_vals, g_vals_smoothed, k_vals, S, S_ideal, prob_ground, prob_excited
