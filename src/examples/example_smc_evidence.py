import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import jax.scipy.special as jsp
import blackjax
import blackjax.smc.resampling as resampling
import blackjax.mcmc.nuts as nuts

SIGMA = 0.1 
DIM = 2

def log_prior_fn(theta):
    return jnp.sum(stats.norm.logpdf(theta, loc=0.0, scale=1.0))

def log_likelihood_fn(theta, data):
    return jnp.sum(stats.norm.logpdf(data, loc=theta, scale=SIGMA))

key = jax.random.key(42)
true_theta = jnp.array([2.0, -1.0])
data = true_theta + SIGMA * jax.random.normal(key, (DIM,))


def compute_evidence_analytic(data, sigma):
    marginal_scale = jnp.sqrt(1.0 + sigma**2)
    return jnp.sum(stats.norm.logpdf(data, loc=0.0, scale=marginal_scale))

Z_analytic = compute_evidence_analytic(data, SIGMA)


def compute_evidence_grid(data, bounds=(-4, 4), grid_size=1000):
    x = jnp.linspace(bounds[0], bounds[1], grid_size)
    y = jnp.linspace(bounds[0], bounds[1], grid_size)
    X, Y = jnp.meshgrid(x, y)
    grid_coords = jnp.stack([X, Y], axis=-1).reshape(-1, 2)
    
    log_priors = jax.vmap(log_prior_fn)(grid_coords)
    log_likes = jax.vmap(lambda t: log_likelihood_fn(t, data))(grid_coords)
    
    cell_area = ((bounds[1] - bounds[0]) / grid_size) ** 2
    return jsp.logsumexp(log_priors + log_likes) + jnp.log(cell_area)

Z_grid = compute_evidence_grid(data)



n_particles = 4000
key, p_key, s_key = jax.random.split(key, 3)
initial_particles = jax.random.normal(p_key, (n_particles, DIM))

mcmc_params = {
    'step_size': jnp.array([1e-2]), 
    'inverse_mass_matrix': jnp.expand_dims(jnp.eye(DIM), 0)
}

smc_loop = blackjax.adaptive_tempered_smc(
    logprior_fn=log_prior_fn,
    loglikelihood_fn=lambda x: log_likelihood_fn(x, data),
    mcmc_step_fn=nuts.build_kernel(),
    mcmc_init_fn=nuts.init,
    mcmc_parameters=mcmc_params,
    resampling_fn=resampling.systematic,
    num_mcmc_steps=30, 
    target_ess=0.8
)

state = smc_loop.init(initial_particles)
step_fn = jax.jit(smc_loop.step)
Z_smc = 0.0
step = 0

while state.tempering_param < 1.0:
    step_key = jax.random.fold_in(s_key, step)
    state, info = step_fn(step_key, state)
    Z_smc += info.log_likelihood_increment
    step += 1
    print(f"Step {step}: Beta={state.tempering_param:.4f}, Total Log Evidence={Z_smc:.4f}")


print(f"\nResults (Target Sigma={SIGMA}):")
print(f"1. Analytical: {Z_analytic:.4f}")
print(f"2. Grid Search: {Z_grid:.4f}")
print(f"3. SMC Est.:    {Z_smc:.4f}")