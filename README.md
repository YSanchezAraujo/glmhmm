# glmhmm: Generalized Linear Model Hidden Markov Models in JAX

`glmhmm` is a Python package for fitting GLM-HMMs using JAX. It allows latent state transitions and emission probabilities to be driven by time-varying external covariates (GLMs), making it ideal for analyzing complex time-series data such as neural recordings or behavioral choices.

## 🚀 Installation

```bash
pip install git+https://github.com/YSanchezAraujo/glmhmm.git
```

## 📊 Data Structure

The core data structure is `GLMHMMData`, which holds lists of arrays. This design supports fitting multiple independent sequences (e.g., different experimental sessions) simultaneously.

### Structure Rules
1.  **Lists of Arrays**: All fields are `List[jnp.ndarray]`, where each element corresponds to one sequence (session).
2.  **Matching Lengths**: For a given sequence $i$, all active fields (transition inputs, emission inputs, observations) must have compatible time dimensions $T_i$.
3.  **Missing Data**: If you don't have data for a specific modality (e.g., no Gaussian data), pass `None`.

### Fields
| Field | Shape per Sequence | Description |
| :--- | :--- | :--- |
| `x_trans` | $(T, D_{trans})$ | Covariates driving state transitions. |
| `x_bern` | $(T, D_{bern})$ | Covariates for Bernoulli emissions. |
| `y_bern` | $(T,)$ | Binary observations (0 or 1). |
| `x_multinom` | $(T, D_{multi})$ | Covariates for Multinomial emissions. |
| `y_multinom` | $(T,)$ | Integer class labels $0 \dots C-1$. |
| `x_negbin` | $(T, D_{nb})$ | Covariates for Negative Binomial emissions. |
| `y_negbin` | $(T,)$ | Integer counts $\ge 0$. |
| `x_gamma` | $(T, D_{gamma})$ | Covariates for Gamma emissions. |
| `y_gamma` | $(T,)$ | Positive continuous values $> 0$. |
| `x_gauss` | $(T_{samples}, D_{gauss})$ | Covariates for Gaussian emissions (e.g. aligned to neural samples). |
| `y_gauss` | $(T_{samples}, M)$ | Continuous multivariate observations. |

> **Note**: For Gaussian data, we support "variable rate" alignment where multiple Gaussian samples map to a single HMM state step. See `gauss_first_idx` / `gauss_last_idx` in `test_fit_compatibility.py` for advanced usage. For standard 1-to-1 mapping, these indices are not needed.

> **Note**: The model is written such that the transition inputs operate on the assumption of $t \rightarrow t+1$. This implies that your transition inputs are aligned to the transition step (i.e. they correctly reflect $t \rightarrow t+1$ and so they will be one less than the total number of observations). 

## 💻 Usage Examples

### 1. Bernoulli GLM-HMM (Binary Choices)
Modeling an agent's binary choices (Left/Right) driven by Bias and Stimulus strength. 

```python
import jax
import jax.numpy as jnp
from glmhmm import fit_glmhmm
from glmhmm.structs import GLMHMMData, GLMHMMConfig
from glmhmm.utils import simulate_data_bernoulli_glmhmm
from glmhmm.m_step import compute_A_from_theta_and_inputs
from glmhmm.fit import e_step

key = jax.random.PRNGKey(42)
N_states = 3
T = 1000
N_bern_features = 2
N_trans_features = 2

# Generate synthetic data
true_params, sim_data = simulate_data_bernoulli_glmhmm(
    key, N_states, T, N_bern_features, N_trans_features
)

# Unwrap simulated data for GLMHMMData
# x_inputs from utils is a list of arrays (one per step), so we concatenate 
# to get a single array of shape (T-1, D) for the session.
x_trans_concat = jnp.concatenate(sim_data.x_inputs, axis=0)

data = GLMHMMData(
    x_bern=[sim_data.x_bern], 
    y_bern=[sim_data.y_bern],
    x_trans=[x_trans_concat] # Transitions driven by inputs (length T-1)
)

config = GLMHMMConfig(
    num_states=N_states,
    num_em_iters=100,
    l2_reg_bern=1.0,    # Regularization for emission weights
    l2_reg_trans=0.5 ,  # Regularization for transition weights
    tol = 1e-6
)


params, lml_history = fit_glmhmm(data, config)

print(f"Converged LML: {lml_history[-1]}")
print("Bernoulli Weights (D x K):")
print(params.W_bern)

# Get the posteriors
gammas, xis, final_lml = e_step(params, data)

# Compute the input-driven transition matrix
A_timevar = compute_A_from_theta_and_inputs(x_trans_concat, params.theta)
```

### 2. Multinomial GLM-HMM (Categorical Choices)
Modeling choices between 3 options (e.g., Left, Right, No-Go).

```python
# X shape: (T, D)
# y shape: (T,) with values {0, 1, 2}

data_multi = GLMHMMData(
    x_multinom=[X_session1],
    y_multinom=[y_session1]
)

config_multi = GLMHMMConfig(
    num_states=2,
    num_em_iters=50,
    l2_reg_multinom=1.0
)

params, _ = fit_glmhmm(data_multi, config_multi)

# W_multinom shape: (D, C, K) 
# The last category (C-1) is fixed to 0 for identifiability.
print(params.W_multinom.shape) 
```

### 3. Model Selection (BIC)
Compare models with different number of states.

```python
from glmhmm.model_selection import compute_bic

for k in [2, 3, 4]:
    cfg = GLMHMMConfig(num_states=k, num_em_iters=100)
    params, h = fit_glmhmm(data, cfg)
    
    # Compute BIC
    bic = compute_bic(data, params, max_log_lik=h[-1])
    print(f"K={k}, BIC={bic:.2f}")
```

## 🛠 Advanced Features

### Factorized Evidence (Bayesian Quadrature)
> **Note**: This is still under construction but for the moment has basic functionality. Also, this works best with continous likelihood models, discrete likelihoods appear to require REALLY LARGE $l^2$ penalties (100 for multinomial for example!). Bernoulli likelihoods seem OK with a modest penalty of 2. 

For rigorous model comparison, use `compute_factorized_evidence` to estimate the marginal likelihood $P(Y)$, integrating over parameter configurations. This is computationally heavier but more accurate than BIC for non-asymptotic regimes.

```python
from glmhmm.fit import e_step
from glmhmm.model_selection import compute_factorized_evidence

# 1. Get posteriors from fitted model
gammas, xis, _ = e_step(params, data)

# 2. Run BQ
log_ev, var = compute_factorized_evidence(
    data, params, config, gammas, xis
)
```
