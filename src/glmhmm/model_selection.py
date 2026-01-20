
import jax.numpy as jnp
from .structs import GLMHMMData, GLMHMMParams

def compute_bic(data: GLMHMMData, params: GLMHMMParams, max_log_lik: float) -> float:
    """
    Computes the Bayesian Information Criterion (BIC) for the fitted model.
    BIC = k * ln(n) - 2 * ln(L)
    
    where:
    k = number of free parameters
    n = number of observations (total time steps) (Wait, strictly BIC uses N observations. For HMM, is it num sequences or num time steps? Usually num time steps if we treat them as samples, though not i.i.d. Standard prac is T.)
    L = maximized value of the likelihood function
    
    Args:
        data: GLMHMMData object
        params: GLMHMMParams object
        max_log_lik: Log-likelihood of the data at the fitted parameters
        
    Returns:
        bic: The BIC value (lower is better)
    """
    n = 0
    if data.x_trans is not None:
         n = sum(x.shape[0] for x in data.x_trans) if isinstance(data.x_trans, list) else data.x_trans.shape[0]
    elif data.x_bern is not None:
         n = sum(x.shape[0] for x in data.x_bern) if isinstance(data.x_bern, list) else data.x_bern.shape[0]
    elif data.x_multinom is not None:
         n = sum(x.shape[0] for x in data.x_multinom) if isinstance(data.x_multinom, list) else data.x_multinom.shape[0]
    elif data.x_negbin is not None:
         n = sum(x.shape[0] for x in data.x_negbin) if isinstance(data.x_negbin, list) else data.x_negbin.shape[0]
    elif data.x_gamma is not None:
         n = sum(x.shape[0] for x in data.x_gamma) if isinstance(data.x_gamma, list) else data.x_gamma.shape[0]
         
    if n == 0: 
        n = 1000 
    
    k = 0
    K = params.log_pi0.shape[0] 
    
    if params.theta is not None:
        D_trans = params.theta.shape[0]
        k += D_trans * K * (K - 1)
    else:
        k += K * (K - 1)
        
    k += (K - 1) 
    
    if params.W_bern is not None:
        k += params.W_bern.size
        
    if params.W_gauss is not None:
        k += params.W_gauss.size
        M = params.W_gauss.shape[1]
        k += K * (M * (M + 1) // 2)
             
    if params.W_multinom is not None:
        D, C, K_check = params.W_multinom.shape
        k += D * (C - 1) * K
        
    if params.W_negbin is not None:
        k += params.W_negbin.size
        if params.phi_negbin is not None:
            k += params.phi_negbin.size
            
    if params.W_gamma is not None:
        k += params.W_gamma.size
        if params.alpha_gamma is not None:
            k += params.alpha_gamma.size
            
    bic = k * jnp.log(n) - 2 * max_log_lik
    return float(bic)

def compute_implied_log_z(bic: float) -> float:
    """
    Computes the implied log marginal likelihood (log Z) from the BIC.
    Approximation: log Z \approx -BIC / 2
    
    Args:
        bic: The Bayesian Information Criterion value.
        
    Returns:
        implied_log_z: The approximated log evidence.
    """
    return -bic / 2.0
