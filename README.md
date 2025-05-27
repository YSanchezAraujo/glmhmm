# Installation

```
pip install git+https://github.com/YSanchezAraujo/glmhmm.git
```

## Details
There is one function `fit_glmhmm` that is exposed in the imported namespace (i.e. `glmhmm.fit_glmhmm`), it fits 3 different models. A Bernoulli emissions GLM-HMM, a Gaussian emissions GLM-HMM and a dual-emissions Bernoulli and Gaussian GLM-HMM. The structure of the dual emissions model is the usual HMM latent sequence with seperate emissions coming out of each latent. Thus the 
factorization is simply `p(Gaussian_t|latent_t) * p(Bernoulli_t|latent_t) * p(latent_t | latent_{t-1})`. If you omit the data in the function call it won't fit that component of the model. 
See the usage example below.

### Data
No mater the type of data you should have some dependent variable `y` and some independent variables in a matrix (where each column is a different variable) `X`. The code assumes a set of these: `y_set = [y_a, y_b, y_c]`, where each `y_a`, `y_b`, `y_c` is itself a vector (Bernoulli) or possibly matrix (Gaussian). The independent variables follow the same structure `x_set = [X_a, X_b, X_c]`, where now each `X_<>` is a matrix. 

If you only have one vector `y_a` and one matrix `X_a` you still need to create variables that contain these in a list, so `y_set = [y_a]` and `x_set = [X_a]`. 

## Example usage
```
import numpy as np
import jax.numpy as jnp
import glmhmm

# set the value of parameters
n_states = 2
dir_diag = 10.0
seed_num = 323
max_iter = 1000
em_tol = 1e-3
ridge_pen = 1.0

# TODO: add simulation functions

model_fit = glmhmm.fit_glmhmm(
    n_states, # n_states
    dir_diag ,# dirichlet diag
    seed_num, # seed
    x_set_bern = x_set,
    y_set_bern = y_set,
    max_iter = 500,
    bern_m_step_maxiter = 500
)
```
