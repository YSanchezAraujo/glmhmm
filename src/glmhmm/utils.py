import jax.numpy as jnp
from collections import namedtuple
from jax import jit, nn, random
from typing import List, Optional, Union
import numpy as np
import itertools

def save_pickle(save_name, obj):
    with open(save_name, "wb") as file:
        pickle.dump(obj, file)

def read_pickle(read_name):
    with open(read_name, "rb") as file:
        return pickle.load(file)

def compute_boundaries(list_of_array):
    """Computes trial/segment info using jax.numpy."""
    n_trials = len(list_of_array)
    samples_per_trial = jnp.array([x.shape[0] for x in list_of_array])
    last_idx = jnp.cumsum(samples_per_trial)
    first_idx = last_idx - samples_per_trial

    ind_info = namedtuple(
        'info', 
        ['n_trials', 'samples_per_trial', 'last_idx', 'first_idx']
        )(n_trials=n_trials, samples_per_trial=samples_per_trial, last_idx=last_idx, first_idx=first_idx)
    
    return ind_info

@jit
def transform_theta(theta, ref_state_ind):
    ref_col = theta[:, :, ref_state_ind]
    ref_col_expanded = ref_col[..., None] 
    theta_prime = theta - ref_col_expanded
    return theta_prime

def compute_A_from_theta_and_inputs(x_trans: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the time-varying transition matrix A_t.
    """
    logits = jnp.einsum('tk,kij->tij', x_trans, theta)
    A = nn.softmax(logits, axis=-1)
    return A

def generate_input_driven_inputs(
    key,
    n_steps,
    n_trans_features,
):
    keys = random.split(key, n_steps)
    x_trans_set = []
    
    for t in range(n_steps - 1):
        x_noise = random.normal(keys[t], (1, n_trans_features - 1))
        x_bias = jnp.ones((1, 1))
        x_trans = jnp.concatenate([x_bias, x_noise], axis=1)
        x_trans_set.append(x_trans)
        
    return x_trans_set


def generate_latent_state_sequence(
    key, 
    n_steps,
    pi0,
    A
):
    keys = random.split(key, n_steps)
    n_states = pi0.shape[0]

    if A.shape[0] == n_states:
        transition_type = "static"
    elif A.shape[0] == n_steps - 1:
        transition_type = "input_driven"
    else:
        raise ValueError("Invalid transition matrix shape.")
    
    state_sequence = jnp.zeros(n_steps, dtype=jnp.int32)
    initial_state = random.choice(keys[0], n_states, p=pi0)
    state_sequence = state_sequence.at[0].set(initial_state)

    for t in range(n_steps - 1):
        latent_t = state_sequence[t]

        if transition_type == "static":
            transition_probs = A[latent_t, :]
        elif transition_type == "input_driven":
            transition_probs = A[t, latent_t, :]
   
        next_state = random.choice(keys[t + 1], n_states, p=transition_probs)
        state_sequence = state_sequence.at[t + 1].set(next_state)

    return state_sequence

def simulate_data_bernoulli_glmhmm(
    key,
    n_states,
    n_hmm_steps,
    n_bern_features,
    n_trans_features,
):
    key, subkey_seq, subkey_trans, subkey_bern = random.split(key, 4)
    pi0 = jnp.array([0.2, 0.3, 0.5])
    
    x_inputs = generate_input_driven_inputs(subkey_trans, n_hmm_steps, n_trans_features)
    x_inputs_concat = jnp.concatenate(x_inputs, axis=0)
    
    key, subkey_trans = random.split(key)
    theta = 0.1 * random.normal(subkey_trans, (n_trans_features, n_states, n_states))
    for i in range(n_states):
        theta = theta.at[0, i, i].add(5.0)

    A = compute_A_from_theta_and_inputs(x_inputs_concat, theta)
    state_sequence = generate_latent_state_sequence(subkey_seq, n_hmm_steps, pi0, A)
    
    W_bern = jnp.zeros((n_bern_features, n_states))
    W_bern = W_bern.at[0, 0].set(-2.0)
    W_bern = W_bern.at[1, 1].set(3.0)
    W_bern = W_bern.at[1, 2].set(-3.0) 

    choices = jnp.zeros(n_hmm_steps, dtype=jnp.int32)
    subkeys_bern = random.split(key, n_hmm_steps)
    x_bern = jnp.zeros((n_hmm_steps, n_bern_features))

    for t in range(n_hmm_steps):
        state = state_sequence[t]
        x_noise = random.normal(subkeys_bern[t], (n_bern_features - 1,))
        x_t = jnp.concatenate([jnp.array([1.0]), x_noise]) 
        
        logit = x_t @ W_bern[:, state]
        prob_y_eq_1 = nn.sigmoid(logit)
        key, subkey_bern = random.split(key)
        choices = choices.at[t].set(random.bernoulli(subkey_bern, prob_y_eq_1))
        x_bern = x_bern.at[t].set(x_t)

    params = namedtuple('params', ['pi0', 'A', 'theta', 'W_bern'])(
        pi0=pi0, A=A, theta=theta, W_bern=W_bern
    )

    data = namedtuple('data', ['state_sequence', 'x_bern', 'y_bern', 'x_inputs'])(
        state_sequence=state_sequence,
        x_bern=x_bern,
        y_bern=choices,
        x_inputs=x_inputs,
    )

    return params, data


def simulate_hmm_states_input_driven(
    key: random.PRNGKey,
    pi0: jnp.ndarray,
    A_t: jnp.ndarray,
    n_hmm_steps: int
) -> jnp.ndarray:
    n_states = pi0.shape[0]
    
    key, subkey_init, subkey_trans = random.split(key, 3)
    state_sequence = jnp.zeros(n_hmm_steps, dtype=jnp.int32)
    
    current_state = random.choice(subkey_init, n_states, p=pi0)
    state_sequence = state_sequence.at[0].set(current_state)

    for t in range(n_hmm_steps - 1):
        transition_probs = A_t[t, current_state, :]
        key, subkey_trans = random.split(key)
        current_state = random.choice(subkey_trans, n_states, p=transition_probs)
        state_sequence = state_sequence.at[t + 1].set(current_state)
        
    return state_sequence


def simulate_data_for_glmhmm(
    key: random.PRNGKey,
    true_params: dict,
    n_sequences: int,
    n_hmm_steps: int,
    # Optional params for Gaussian data
    n_gauss_features: Optional[int] = None,
    n_regions: Optional[int] = None,
    samples_per_gaussian_segment: Optional[Union[int, List[int]]] = None,
    # Optional params for Bernoulli data
    n_bern_features: Optional[int] = None,
    # Optional params for transition data
    # Optional params for transition data
    n_trans_features: Optional[int] = None,
    # Optional params for new emissions
    n_multinom_features: Optional[int] = None,
    n_multinom_choices: Optional[int] = None,
    n_negbin_features: Optional[int] = None,
    n_gamma_features: Optional[int] = None
) -> dict:
    """
    Simulates data for the general fit_glmhmm function.
    Can simulate for Bernoulli-only, Gaussian-only, or joint models,
    with static or input-driven transitions.
    """
    true_pi0 = true_params["pi0"]
    true_A = true_params.get("A")
    true_theta = true_params.get("theta")
    true_W_bern = true_params.get("W_bern")
    true_W_gauss = true_params.get("W_gauss")
    true_W_gauss = true_params.get("W_gauss")
    true_Sigma_gauss = true_params.get("Sigma_gauss")
    true_W_multinom = true_params.get("W_multinom")
    true_W_negbin = true_params.get("W_negbin")
    true_phi_negbin = true_params.get("phi_negbin")
    true_W_gamma = true_params.get("W_gamma")
    true_alpha_gamma = true_params.get("alpha_gamma")

    has_bern = true_W_bern is not None
    has_gauss = true_W_gauss is not None and true_Sigma_gauss is not None
    has_multinom = true_W_multinom is not None
    has_negbin = true_W_negbin is not None and true_phi_negbin is not None
    has_gamma = true_W_gamma is not None and true_alpha_gamma is not None
    is_input_driven = true_theta is not None

    all_state_sequences = []
    all_x_trans_set = []

    for i in range(n_sequences):
        key, subkey_seq = random.split(key)
        if is_input_driven:
            x_trans = generate_input_driven_inputs(subkey_seq, n_hmm_steps, n_trans_features)
            x_trans_concat = jnp.concatenate(x_trans, axis=0)
            all_x_trans_set.append(x_trans_concat)
            
            A_t = compute_A_from_theta_and_inputs(x_trans_concat, true_theta)
            states = simulate_hmm_states_input_driven(subkey_seq, true_pi0, A_t, n_hmm_steps)
        else:
            states = simulate_hmm_states_input_driven(subkey_seq, true_pi0, jnp.tile(true_A, (n_hmm_steps - 1, 1, 1)), n_hmm_steps)
        
        all_state_sequences.append(states)

    all_x_bern_set, all_y_bern_set = [], []
    all_x_bern_set, all_y_bern_set = [], []
    all_x_multinom_set, all_y_multinom_set = [], []
    all_x_negbin_set, all_y_negbin_set = [], []
    all_x_gamma_set, all_y_gamma_set = [], []
    all_x_gauss_nested_set, all_y_gauss_nested_set = [], []

    for i in range(n_sequences):
        key, subkey_seq_obs = random.split(key)
        states_for_seq = all_state_sequences[i]
        
        if has_bern:
            x_noise = random.normal(subkey_seq_obs, (n_hmm_steps, n_bern_features - 1))
            x_bias = jnp.ones((n_hmm_steps, 1))
            x_bern_seq = jnp.concatenate([x_bias, x_noise], axis=1)
            
            y_bern_seq = jnp.zeros(n_hmm_steps, dtype=jnp.int32)
            for t in range(n_hmm_steps):
                key, subkey_obs = random.split(key)
                state = states_for_seq[t]
                logit = x_bern_seq[t, :] @ true_W_bern[:, state]
                prob_y_eq_1 = nn.sigmoid(logit)
                y_bern_seq = y_bern_seq.at[t].set(random.bernoulli(subkey_obs, prob_y_eq_1))
            all_x_bern_set.append(x_bern_seq)
            all_y_bern_set.append(y_bern_seq)

        if has_gauss:
            x_gauss_nested_trials, y_gauss_nested_trials = [], []
            if isinstance(samples_per_gaussian_segment, int):
                current_samples_per_segment = [samples_per_gaussian_segment] * n_hmm_steps
            else:
                current_samples_per_segment = samples_per_gaussian_segment

            for t in range(n_hmm_steps):
                key, subkey_x, subkey_y = random.split(key, 3)
                state = states_for_seq[t]
                n_samples_this_segment = current_samples_per_segment[t]
                x_gauss_trial = random.normal(subkey_x, (n_samples_this_segment, n_gauss_features))
                mu_gauss_trial = x_gauss_trial @ true_W_gauss[:, :, state]
                y_gauss_trial = random.multivariate_normal(subkey_y, mean=mu_gauss_trial, cov=true_Sigma_gauss[:, :, state])
                
                x_gauss_nested_trials.append(x_gauss_trial)
                y_gauss_nested_trials.append(y_gauss_trial)

            all_x_gauss_nested_set.append(x_gauss_nested_trials)
            all_y_gauss_nested_set.append(y_gauss_nested_trials)

        if has_multinom:
            key, subkey_mn = random.split(key)
            x_noise = random.normal(subkey_mn, (n_hmm_steps, n_multinom_features - 1))
            x_bias = jnp.ones((n_hmm_steps, 1))
            x_mn_seq = jnp.concatenate([x_bias, x_noise], axis=1) # (T, D)

            y_mn_seq = jnp.zeros(n_hmm_steps, dtype=jnp.int32)
            for t in range(n_hmm_steps):
                key, subkey_obs = random.split(key)
                state = states_for_seq[t]
                logits = x_mn_seq[t] @ true_W_multinom[:, :, state]
                probs = nn.softmax(logits)
                y_mn_seq = y_mn_seq.at[t].set(random.categorical(subkey_obs, jnp.log(probs)))
            
            all_x_multinom_set.append(x_mn_seq)
            all_y_multinom_set.append(y_mn_seq)

        if has_negbin:
            key, subkey_nb = random.split(key)
            x_noise = random.normal(subkey_nb, (n_hmm_steps, n_negbin_features - 1))
            x_bias = jnp.ones((n_hmm_steps, 1))
            x_nb_seq = jnp.concatenate([x_bias, x_noise], axis=1) # (T, D)

            y_nb_seq = jnp.zeros(n_hmm_steps, dtype=jnp.float32)
            for t in range(n_hmm_steps):
                key, subkey_obs, subkey_pois = random.split(key, 3)
                state = states_for_seq[t]
                mu = jnp.exp(x_nb_seq[t] @ true_W_negbin[:, state])
                phi = true_phi_negbin[state]
                gamma_sample = random.gamma(subkey_obs, phi) * (mu / phi)
                y_nb_seq = y_nb_seq.at[t].set(random.poisson(subkey_pois, gamma_sample))
            
            all_x_negbin_set.append(x_nb_seq)
            all_y_negbin_set.append(y_nb_seq)

        if has_gamma:
            key, subkey_gm = random.split(key)
            x_noise = random.normal(subkey_gm, (n_hmm_steps, n_gamma_features - 1))
            x_bias = jnp.ones((n_hmm_steps, 1))
            x_gm_seq = jnp.concatenate([x_bias, x_noise], axis=1) # (T, D)

            y_gm_seq = jnp.zeros(n_hmm_steps, dtype=jnp.float32)
            for t in range(n_hmm_steps):
                key, subkey_obs = random.split(key)
                state = states_for_seq[t]
                mu = jnp.exp(x_gm_seq[t] @ true_W_gamma[:, state])
                alpha = true_alpha_gamma[state]
                gamma_sample = random.gamma(subkey_obs, alpha) * (mu / alpha)
                y_gm_seq = y_gm_seq.at[t].set(gamma_sample)
                
            all_x_gamma_set.append(x_gm_seq)
            all_y_gamma_set.append(y_gm_seq)

    simulated_data = {
        "x_trans_set": all_x_trans_set if is_input_driven else None,
        "x_set_bern": all_x_bern_set if has_bern else None,
        "y_set_bern": all_y_bern_set if has_bern else None,
        "x_multinom": all_x_multinom_set if has_multinom else None,
        "y_multinom": all_y_multinom_set if has_multinom else None,
        "x_negbin": all_x_negbin_set if has_negbin else None,
        "y_negbin": all_y_negbin_set if has_negbin else None,
        "x_gamma": all_x_gamma_set if has_gamma else None,
        "y_gamma": all_y_gamma_set if has_gamma else None,
        "x_nested_set": all_x_gauss_nested_set if has_gauss else None,
        "y_nested_set": all_y_gauss_nested_set if has_gauss else None,
        "true_states": all_state_sequences,
        "true_params": true_params
    }
    return simulated_data


def find_best_permutation(true_params, fit_output):
    """
    Finds the state permutation that minimizes the MSE between true and fitted parameters.
    Handles 'base class' alignment for transition weights (theta) automatically.
    
    Args:
        true_params (dict): Dictionary containing true parameters.
        fit_output (namedtuple): Output from fit_glmhmm containing fitted parameters.
        
    Returns:
        best_perm (tuple): The optimal order of states (e.g., (2, 0, 1))
        permuted_fit (dict): A dictionary with the fitted parameters re-ordered to match truth.
    """
    
    has_bern = hasattr(fit_output, 'W_bern') and fit_output.W_bern is not None and 'W_bern' in true_params and true_params['W_bern'] is not None
    has_input_trans = hasattr(fit_output, 'theta') and fit_output.theta is not None and 'theta' in true_params and true_params['theta'] is not None
    has_static_trans = hasattr(fit_output, 'log_A') and fit_output.log_A is not None and 'A' in true_params and true_params['A'] is not None
    has_multinom = hasattr(fit_output, 'W_multinom') and fit_output.W_multinom is not None and 'W_multinom' in true_params and true_params['W_multinom'] is not None
    has_negbin = hasattr(fit_output, 'W_negbin') and fit_output.W_negbin is not None and 'W_negbin' in true_params and true_params['W_negbin'] is not None
    has_phi_nb = hasattr(fit_output, 'phi_negbin') and fit_output.phi_negbin is not None and 'phi_negbin' in true_params and true_params['phi_negbin'] is not None
    has_gamma = hasattr(fit_output, 'W_gamma') and fit_output.W_gamma is not None and 'W_gamma' in true_params and true_params['W_gamma'] is not None
    has_alpha_gm = hasattr(fit_output, 'alpha_gamma') and fit_output.alpha_gamma is not None and 'alpha_gamma' in true_params and true_params['alpha_gamma'] is not None
    has_gauss = hasattr(fit_output, 'W_gauss') and fit_output.W_gauss is not None and 'W_gauss' in true_params and true_params['W_gauss'] is not None

    n_states = 0
    if has_bern:
        n_states = true_params['W_bern'].shape[1]
    elif has_input_trans:
        n_states = true_params['theta'].shape[1]
    elif has_static_trans:
        n_states = true_params['A'].shape[0]
    elif has_multinom:
        n_states = true_params['W_multinom'].shape[2] # (F, C, K)
    
    if n_states == 0:
         raise ValueError("Could not determine n_states or find matching parameters.")

    def align_last_dim_zero(w):
        pass

    def align_trans_weights(w):
        base_col = w[:, :, -1:] 
        return w - base_col

    best_perm = None
    best_mse = float('inf')
    
    t_W_b = np.array(true_params['W_bern']) if has_bern else None
    f_W_b = np.array(fit_output.W_bern) if has_bern else None
    
    t_theta = np.array(true_params['theta']) if has_input_trans else None
    f_theta = np.array(fit_output.theta) if has_input_trans else None
    
    t_A = np.array(true_params['A']) if has_static_trans else None
    f_log_A = np.array(fit_output.log_A) if has_static_trans else None
    f_A = np.exp(f_log_A) if has_static_trans else None
    
    t_W_mn = np.array(true_params['W_multinom']) if has_multinom else None
    f_W_mn = np.array(fit_output.W_multinom) if has_multinom else None
    
    t_W_nb = np.array(true_params['W_negbin']) if has_negbin else None
    f_W_nb = np.array(fit_output.W_negbin) if has_negbin else None
    
    t_phi = np.array(true_params['phi_negbin']) if has_phi_nb else None
    f_phi = np.array(fit_output.phi_negbin) if has_phi_nb else None
    
    t_W_gm = np.array(true_params['W_gamma']) if has_gamma else None
    f_W_gm = np.array(fit_output.W_gamma) if has_gamma else None
    
    t_alpha = np.array(true_params['alpha_gamma']) if has_alpha_gm else None
    f_alpha = np.array(fit_output.alpha_gamma) if has_alpha_gm else None
    
    t_W_gs = np.array(true_params['W_gauss']) if has_gauss else None
    f_W_gs = np.array(fit_output.W_gauss) if has_gauss else None
    t_Sig = np.array(true_params['Sigma_gauss']) if has_gauss else None
    f_Sig = np.array(fit_output.Sigma_gauss) if has_gauss else None
    
    
    for p_tuple in itertools.permutations(range(n_states)):
        p = list(p_tuple)
        current_mse = 0.0
        count = 0
        
        if has_bern:
            perm_W_b = f_W_b[:, p]
            current_mse += np.mean((perm_W_b - t_W_b)**2)
            count += 1
            
        if has_input_trans:
            perm_theta = f_theta[:, p, :][:, :, p]
            t_aligned = align_trans_weights(t_theta)
            p_aligned = align_trans_weights(perm_theta)
            current_mse += np.mean((p_aligned - t_aligned)**2)
            count += 1
            
        if has_static_trans:
            perm_A = f_A[p, :][:, p]
            current_mse += np.mean((perm_A - t_A)**2)
            count += 1
            
        if has_multinom:
            perm_W_mn = f_W_mn[:, :, p]
            p_mn_aligned = perm_W_mn - perm_W_mn[:, -1:, :]
            t_mn_aligned = t_W_mn - t_W_mn[:, -1:, :]
            
            current_mse += np.mean((p_mn_aligned - t_mn_aligned)**2)
            count += 1
            
        if has_negbin:
            perm_W_nb = f_W_nb[:, p]
            current_mse += np.mean((perm_W_nb - t_W_nb)**2)
            count += 1
            
        if has_phi_nb:
            perm_phi = f_phi[p]
            current_mse += np.mean((perm_phi - t_phi)**2)
            count += 1

        if has_gamma:
            perm_W_gm = f_W_gm[:, p]
            current_mse += np.mean((perm_W_gm - t_W_gm)**2)
            count += 1
            
        if has_alpha_gm:
            perm_alpha = f_alpha[p]
            current_mse += np.mean((perm_alpha - t_alpha)**2)
            count += 1
            
        if has_gauss:
            perm_W_gs = f_W_gs[:, :, p]
            current_mse += np.mean((perm_W_gs - t_W_gs)**2)
            perm_Sig = f_Sig[:, :, p]
            current_mse += np.mean((perm_Sig - t_Sig)**2)
            count += 1
            
        if current_mse < best_mse:
            best_mse = current_mse
            best_perm = p

    print(f"Best Permutation found: {best_perm} with MSE: {best_mse:.4f}")
    
    permuted_results = {}
    p = best_perm
    
    if has_bern: 
        permuted_results['W_bern'] = f_W_b[:, p]
    if has_input_trans:
        pt = f_theta[:, p, :][:, :, p]
        permuted_results['theta'] = align_trans_weights(pt)
    if has_static_trans:
        pA = f_A[p, :][:, p]
        permuted_results['A'] = pA
        permuted_results['log_A'] = np.log(pA + 1e-9)
    if has_multinom:
        p_mn = f_W_mn[:, :, p]
        permuted_results['W_multinom'] = p_mn - p_mn[:, -1:, :]
    if has_negbin: 
        permuted_results['W_negbin'] = f_W_nb[:, p]
    if has_phi_nb: 
        permuted_results['phi_negbin'] = f_phi[p]
    if has_gamma: 
        permuted_results['W_gamma'] = f_W_gm[:, p]
    if has_alpha_gm: 
        permuted_results['alpha_gamma'] = f_alpha[p]
    if has_gauss:
        permuted_results['W_gauss'] = f_W_gs[:, :, p]
        permuted_results['Sigma_gauss'] = f_Sig[:, :, p]
        
    return best_perm, permuted_results

def simulate_train_test_data(
    key, 
    K=3, 
    D=2, 
    n_train=10, 
    n_test=5, 
    n_steps=200
):
    """
    Generates synthetic data for Bernoulli GLM-HMM with static transitions.
    Returns:
        train_data: GLMHMMData
        test_data: GLMHMMData
        full_data: GLMHMMData (concatenation of train and test)
        true_params: dict
    """
    key, subkey_p = random.split(key)
    
    if K == 3:
        A = jnp.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9]
        ])
    else:
        A = jnp.eye(K) * 0.8 + 0.2/K
        A = A / jnp.sum(A, axis=1, keepdims=True)
        
    pi0 = jnp.ones(K) / K
    
    key, subkey_w = random.split(key)
    W = random.normal(subkey_w, (D, K)) 
    W = W.at[0, 0].set(0.0) 

    W = W.at[1, 1].set(4.0) 
    W = W.at[0, 1].set(0.0)

    W = W.at[1, 2].set(0.0)
    W = W.at[0, 2].set(3.0)
        
    true_params = {
        "pi0": pi0,
        "A": A,
        "W_bern": W
    }
    
    key, subkey_train = random.split(key)
    train_sim = simulate_data_for_glmhmm(
        subkey_train, true_params, n_train, n_steps, n_bern_features=D
    )
    
    key, subkey_test = random.split(key)
    test_sim = simulate_data_for_glmhmm(
        subkey_test, true_params, n_test, n_steps, n_bern_features=D
    )
    
    from .structs import GLMHMMData
    
    train_data = GLMHMMData(
        x_bern=train_sim['x_set_bern'],
        y_bern=train_sim['y_set_bern']
    )
    
    test_data = GLMHMMData(
        x_bern=test_sim['x_set_bern'],
        y_bern=test_sim['y_set_bern']
    )
    
    full_data = GLMHMMData(
        x_bern=train_sim['x_set_bern'] + test_sim['x_set_bern'],
        y_bern=train_sim['y_set_bern'] + test_sim['y_set_bern']
    )
    
    return train_data, test_data, full_data, true_params