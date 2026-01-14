import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from glmhmm import fit_glmhmm
from glmhmm.utils import simulate_data_for_glmhmm
import numpy as np
plt.ion()

key = jax.random.PRNGKey(42)
n_states = 3
n_steps = 5000  # Increased slightly to ensure enough transitions occur
n_bern_features = 2   # [Bias, Stimulus]
n_trans_features = 2  # [Bias, Fatigue]

# A. True Emissions (W_bern)
# State 0 (Lazy): Negative bias, ignores stimulus
# State 1 (Worker): Neutral bias, strong positive response to stimulus
# State 2 (Rebel): Positive bias, strong negative response to stimulus
W_bern_true = jnp.array([
    [-2.0,  0.0,  1.0],  # Bias (Row 0)
    [ 0.0,  3.0, -3.0]   # Stimulus (Row 1)
])

# B. True Transitions (Theta)
# Shape: (Features, From, To)
theta_true = jnp.zeros((n_trans_features, n_states, n_states))

# 1. Bias Row (Row 0): Sticky States
# We add +4.0 to diagonal to make states persist ~98% of the time
sticky_strength = 4.0 
for k in range(n_states):
    theta_true = theta_true.at[0, k, k].set(sticky_strength) 

# 2. Input Trigger (Row 1): Fatigue
# Feature 1 drives transition from State 1 -> State 0
theta_true = theta_true.at[1, 1, 0].set(3.0) 
# Feature 1 drives transition from State 2 -> State 0
theta_true = theta_true.at[1, 2, 0].set(3.0)
#theta_true = theta_true.at[0, 2, 2].set(0.0)

true_params = {
    "pi0": jnp.ones(n_states) / n_states,
    "theta": theta_true,
    "W_bern": W_bern_true,
    "W_gauss": None,
    "Sigma_gauss": None
}

# --- 2. SIMULATE DATA ---
print("Simulating Data...")
# This now generates inputs with the Bias column (1.0) already included
sim_data = simulate_data_for_glmhmm(
    key, true_params, 
    n_sequences=1, 
    n_hmm_steps=n_steps,
    n_bern_features=n_bern_features, 
    n_trans_features=n_trans_features
)

# Unpack data directly (No manual fix needed!)
x_trans = sim_data["x_trans_set"]
x_bern = sim_data["x_set_bern"]
y_bern = sim_data["y_set_bern"]
true_states = sim_data["true_states"][0]

print(f"Simulated {n_steps} steps.")
print(f"True State Counts: {jnp.bincount(true_states)}")

# --- 3. FIT THE MODEL ---
print("\nFitting GLM-HMM...")
dir_diag = jnp.array([10.0, 10.0, 10.0])

fit_output = fit_glmhmm(
    n_states=n_states,
    dir_diag=jnp.ones(n_states),
    seed=3234,
    transition_type="input_driven",
    
    # Inputs (Now correct directly from simulation)
    x_trans_set=x_trans,
    x_set_bern=x_bern,
    y_set_bern=y_bern,
    
    # Disable Gaussian
    x_nested_set=None,
    y_nested_set=None,
    segment_gaussian=False,
    
    # CRITICAL HYPERPARAMS FOR RECOVERY
    max_iter=500,
    ridge_pen=0.1,           # Penalty for W_bern is fine
    transition_ridge_pen=0.005,# Turn OFF penalty for transitions to recover sparse signals
    sticky_bias=1.0,         # Set to 1.0 (not 10.0) to allow state switching
    tol_optax=1e-5
)

