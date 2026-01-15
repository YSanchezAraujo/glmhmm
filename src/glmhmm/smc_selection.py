import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from jax import flatten_util
from functools import partial
import blackjax
import blackjax.smc.resampling as resampling
import blackjax.mcmc.nuts as nuts

from .structs import GLMHMMData, GLMHMMParams, GLMHMMConfig
from .fit import compute_log_marginal_likelihood

def transform_params_to_unconstrained(params: GLMHMMParams):
    """
    Transforms constrained GLMHMM parameters to an unconstrained dictionary.
    
    Supported:
    - log_pi0 -> logits_pi0 (via identity, since log_pi0 is already log-prob, we treat it as logits for softmax)
    - log_A -> logits_A
    - W_bern -> W_bern (already unconstrained)
    """
    unconstrained = {}
    unconstrained["logits_pi0"] = params.log_pi0 
    unconstrained["logits_A"] = params.log_A
    
    if params.W_bern is not None:
        unconstrained["W_bern"] = params.W_bern
        
    return unconstrained

def transform_unconstrained_to_params(unconstrained: dict, prototypes: GLMHMMParams) -> GLMHMMParams:
    """
    Transforms unconstrained dictionary back to GLMHMMParams.
    """
    logits_pi = unconstrained["logits_pi0"]
    log_pi0 = jax.nn.log_softmax(logits_pi)
    
    logits_A = unconstrained["logits_A"]
    log_A = jax.nn.log_softmax(logits_A, axis=-1)
    
    W_bern = None
    if "W_bern" in unconstrained:
        W_bern = unconstrained["W_bern"]
        
    return GLMHMMParams(
        log_pi0=log_pi0,
        log_A=log_A,
        theta=None,
        W_bern=W_bern,
        W_multinom=None,
        W_negbin=None,
        phi_negbin=None,
        W_gamma=None,
        alpha_gamma=None,
        W_gauss=None,
        Sigma_gauss=None
    )

def log_prior_fn(flat_params, unflatten_fn, prototype_params, config: GLMHMMConfig):
    """
    Calculates log prior for flattened parameters.
    Assumes standard Gaussian priors on unconstrained parameters.
    """
    unconstrained_pytree = unflatten_fn(flat_params)
    
    log_p = 0.0
    log_p += jnp.sum(stats.norm.logpdf(unconstrained_pytree["logits_pi0"], 0.0, 1.0))
    log_p += jnp.sum(stats.norm.logpdf(unconstrained_pytree["logits_A"], 0.0, 1.0))

    if "W_bern" in unconstrained_pytree:
        w_bern = unconstrained_pytree["W_bern"] 
        l2_lambda = config.l2_reg_bern
        
        if l2_lambda > 0:
            scale = 1.0 / jnp.sqrt(l2_lambda)
            log_p += jnp.sum(stats.norm.logpdf(w_bern[1:], 0.0, scale))
        else:
            log_p += jnp.sum(stats.norm.logpdf(w_bern[1:], 0.0, 10.0))
            
        log_p += jnp.sum(stats.norm.logpdf(w_bern[0], 0.0, 10.0))
        
    return log_p

def log_likelihood_fn(flat_params, unflatten_fn, prototype_params, data):
    """
    Calculates log likelihood for flattened parameters.
    """
    unconstrained_pytree = unflatten_fn(flat_params)
    params = transform_unconstrained_to_params(unconstrained_pytree, prototype_params)
    
    return compute_log_marginal_likelihood(params, data)

@partial(jax.jit, static_argnames=["smc_loop"])
def fast_smc_evidence(smc_loop, initial_particles, key):
    initial_state = smc_loop.init(initial_particles)
    
    init_val = (key, initial_state, 0.0)

    def cond_fun(val):
        _, state, _ = val
        return state.tempering_param < 1.0

    def body_fun(val):
        key, state, evidence = val
        key, step_key = jax.random.split(key)
        
        new_state, info = smc_loop.step(step_key, state)
        
        new_evidence = evidence + info.log_likelihood_increment
        return (key, new_state, new_evidence)

    final_val = jax.lax.while_loop(cond_fun, body_fun, init_val)
    
    _, _, total_log_evidence = final_val
    return total_log_evidence

def run_smc_evidence_estimation(
    data: GLMHMMData,
    prototype_params: GLMHMMParams,
    config: GLMHMMConfig,
    num_particles: int = 1000,
    num_mcmc_steps: int = 20,
    key_seed: int = 42
):
    """
    Estimates the model evidence P(Data) using Adaptive Tempered SMC.
    
    Args:
        data: GLMHMMData object.
        prototype_params: An instance of GLMHMMParams to define the structure (shapes).
                          Values in this param object are used as initial guess or shape prototypes.
        config: GLMHMMConfig object containing regularization parameters for priors.
        num_particles: Number of SMC particles.
        num_mcmc_steps: Number of MCMC steps per SMC tempering stage.
        key_seed: Random seed.
        
    Returns:
        log_evidence (float): Estimated log marginal likelihood.
    """
    key = jax.random.PRNGKey(key_seed)
    
    init_unconstrained = transform_params_to_unconstrained(prototype_params)
    flat_init, unflatten_fn = flatten_util.ravel_pytree(init_unconstrained)
    dim = flat_init.shape[0]
    
    prior = partial(log_prior_fn, unflatten_fn=unflatten_fn, prototype_params=prototype_params, config=config)
    likelihood = partial(log_likelihood_fn, unflatten_fn=unflatten_fn, prototype_params=prototype_params, data=data)
    
    key, init_key = jax.random.split(key)
    initial_particles = jax.random.normal(init_key, (num_particles, dim))
    
    mcmc_params = {
        'step_size': jnp.array([1e-2]), 
        'inverse_mass_matrix': jnp.expand_dims(jnp.eye(dim), 0)
    }

    smc_loop = blackjax.adaptive_tempered_smc(
        logprior_fn=prior,
        loglikelihood_fn=likelihood,
        mcmc_step_fn=nuts.build_kernel(),
        mcmc_init_fn=nuts.init,
        mcmc_parameters=mcmc_params,
        resampling_fn=resampling.systematic,
        num_mcmc_steps=num_mcmc_steps, 
        target_ess=0.5
    )
    
    print("Running JIT-compiled SMC loop... (this may take a moment to compile)")
    log_evidence = fast_smc_evidence(smc_loop, initial_particles, key)

    return float(log_evidence)
