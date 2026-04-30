
import matplotlib.pyplot as plt
from qutip import *
import numpy as np
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter1d

### SET UP FOR A THREE LEVEL SYSTEM
ground = Qobj([[1],[0],[0]])  
storage = Qobj([[0],[1],[0]]) 
excited = Qobj([[0],[0],[1]]) 

sigma_ee = Qobj([[0,0,0],[0,0,0],[0,0,1]])  # |e><e| (excited state population)
sigma_ss = Qobj([[0,0,0],[0,1,0],[0,0,0]])  # |s><s| (storage state population)
sigma_gg = Qobj([[1,0,0],[0,0,0],[0,0,0]])  # |g><g| (ground state population)

sigma_ge = Qobj([[0,0,0],[0,0,0],[1,0,0]])  # |e><g| (transition from g to e)
sigma_eg = Qobj([[0,0,1],[0,0,0],[0,0,0]])  # |g><e| (transition from e to g)

sigma_se = Qobj([[0,0,0],[0,0,0],[0,1,0]])  # |e><s| (transition from s to e)
sigma_es = Qobj([[0,0,0],[0,0,1],[0,0,0]])  # |s><e| (transition from e to s)

sigma_gs = Qobj([[0,0,0],[1,0,0],[0,0,0]])  # |s><g| (transition from g to s)
sigma_sg = Qobj([[0,1,0],[0,0,0],[0,0,0]])  # |g><s| (transition from s to g)


def three_level_simulation(C, gamma, Delta, time_final, T, N, g_se, g_ge, alpha_in_func, template_func):
    time = np.linspace(0, time_final, N)
    alpha = alpha_in_func(time)
    template = template_func(time)
    
    # --------------------------- CALCULATE BETA(T) ---------------------------
    # Prefactors
    a = -(gamma*(1+C) - 1j*Delta) / np.sqrt(2*gamma*(1+C))
    c = gamma**2*(1+C)**2 + Delta**2

    # Compute b
    b = [0]
    for i in range(1, len(time)):
        integral_val = integrate.simps(abs(template[0:i])**2, time[0:i]) # integral (0 to t) of |template|^2
        if (integral_val == 0):
            b.append(0)
        else:      
            b.append(template[i] / (np.sqrt(integral_val)))
    b = np.array(b)
    
    # Function to get h(t, T) - returns the integral of |beta(t)|^2 from t[idx] to t[T_index]
    def get_h(beta_vals, idx, time):
        T_index = np.where(time > T)[0][0] # find index corresponding to time T (closest)
        if idx >= T_index:
            h_val = 0
        else:
            h_val = integrate.simps(np.abs(beta_vals[idx:T_index])**2, time[idx:T_index]) # Integrate beta from time[idx] to time[T_index]
        return h_val    

    beta_vals = np.array([0])
    for idx in range(len(time)-1):
        h = get_h(a*b, idx, time)
        beta_vals = np.append(beta_vals, a*b[idx]*np.exp(1j*Delta*h/c))
    
    # Returns interpolated value of beta at time t
    def beta(t):
        if t == time[0]: return 0
        else: return np.interp(t, time, beta_vals)
    
    # --------------------------- HAMILTONIAN ----------------------------
    H_signal = sigma_ge
    H_signal_dag = sigma_eg
    H_coupling = sigma_se
    H_coupling_dag = sigma_es
    H_detuning = sigma_ee 
    
    def H_signal_coeff(t, args): 
        return g_ge * alpha_in_func(t) 

    def H_signal_dag_coeff(t, args): 
        return g_ge * np.conjugate(alpha_in_func(t))

    def H_coupling_coeff(t, args):
        return g_se * beta(t)

    def H_coupling_dag_coeff(t, args):
        return g_se * np.conjugate(beta(t))

    H = [[H_detuning, Delta], [H_signal, H_signal_coeff], [H_signal_dag, H_signal_dag_coeff], 
                              [H_coupling, H_coupling_coeff], [H_coupling_dag, H_coupling_dag_coeff]] 

    # --------------------------- DISSIPATION ---------------------------
    c_op_list = [[sigma_eg, -np.conjugate(g_ge)]] 
                 #[sigma_es, -g_se]]  # with decay into storage state
    
    # --------------------------- SIMULATE ---------------------------
    psi0 = ground
    result = mesolve(H, psi0, time, c_op_list)
    
    # --------------------------- EXTRACT RESULTS ---------------------------
    prob_g = []
    prob_s = []
    prob_e = []
    S = []
    for state in result.states:
        p_g, p_s, p_e = state[0][0], state[1][1], state[2][2]
        prob_g.append(p_g)
        prob_s.append(p_s)
        prob_e.append(p_e)
        S.append(expect(sigma_gs, state))
    
    return time, np.array(S), np.array(prob_s), np.array(prob_g), np.array(prob_e), alpha, beta_vals


# PLOT THE PROBABILITIES IN EACH STATE AND STORAGE S
def plot_probs(time, prob_g, prob_e, prob_s, S, input_photon_number):
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    axs[0].plot(time, prob_g + prob_e + prob_s, color='red', label='Total')
    axs[0].plot(time, prob_g, color='blue', label='Ground')
    axs[0].set_ylabel('Probability')
    axs[0].set_title('Ground State Probability')
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(time, prob_s, color='orange')
    axs[1].plot(time, input_photon_number*np.ones(len(time)), color='orange', linestyle='--', label='Limit')
    axs[1].set_ylabel('Probability')
    axs[1].set_title('Storage State Probability')
    axs[1].grid()

    axs[2].plot(time, prob_e, color='green')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Probability')
    axs[2].set_title('Excited State Probability')
    axs[2].grid()
    
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(time, abs(S), label='|S|')
    plt.plot(time, input_photon_number**0.5*np.ones(len(time)), color='black', linestyle='--', label='Limit')
    plt.plot(time, np.real(S), label='Re(S)')
    plt.plot(time, np.imag(S), label='Im(S)')
    plt.xlabel('Time')
    plt.ylabel('S')
    plt.legend()
    plt.grid()
    plt.show()