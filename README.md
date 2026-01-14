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

## 💻 Usage Examples

### 1. Bernoulli GLM-HMM (Binary Choices)
Modeling an agent's binary choices (Left/Right) driven by Bias and Stimulus strength.

```python
import jax
import jax.numpy as jnp
from glmhmm import fit_glmhmm
from glmhmm.structs import GLMHMMData, GLMHMMConfig

# --- 1. Prepare Data ---
# Suppose we have 2 sessions.
# Session 1: 500 trials
T1 = 500
X1 = jnp.zeros((T1, 2)) # [Bias=1, Stimulus]
y1 = jnp.zeros(T1)      # Choices

# Session 2: 400 trials
T2 = 400
X2 = jnp.zeros((T2, 2))
y2 = jnp.zeros(T2)

# Wrap in lists
data = GLMHMMData(
    x_bern=[X1, X2], 
    y_bern=[y1, y2],
    # We can also add transition inputs:
    x_trans=[X1, X2] # Transitions driven by same inputs
)

# --- 2. Configure ---
config = GLMHMMConfig(
    num_states=3,
    num_em_iters=100,
    l2_reg_bern=1.0,   # Regularization for emission weights
    l2_reg_trans=0.5   # Regularization for transition weights
)

# --- 3. Fit ---
params, lml_history = fit_glmhmm(data, config)

print(f"Converged LML: {lml_history[-1]}")
print("Bernoulli Weights (D x K):")
print(params.W_bern)
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
*NOTE:* This works best with continous likelihood models, discrete likelihoods appear to require REALLY LARGE $l^2$ penalties (100 for multinomial for example!). Bernoulli likelihoods seem OK with a modest penalty of 2. 

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
