This contains code for RbQ simulations, written with Qutip.

transition_amplitudes --> transition_matrix_elements.ipynb: Calculated Zeeman energy shifts for the 2S_1/2, 2P_1/2, 2P_3/2 states. Produces dipole transition moment over changing magnetic field but the scaling is off still. 

Two Level System FREESPACE Simulations: 
- two_level_freespace_walkthrough: In-depth walkthrough of the simulation (most up-to-date)
- two_level_summary: Shortened two_level_freespace_walkthrough 
- two_level_funcs/two_level_notebook: for external user use, converts to function that returns simulation data to Jupyter notebook
- Issues: alpha_out is off by some phase

Two Level System CAVITY Simulations: 
- two_level_with_cavity: functional (issue with scaling kappa term)

Three Level System FREESPACE Simulations: 
- three_level_freespace_walkthrough: functional (slight issues with the efficiency being higher, units)
- three_level_freespace_notebook/three_level_freespace_funcs: written to allow user to input an arbitrary pulse

Three Level System CAVITY Simulations
- three_level_cavity: not functional yet
