import torch
from torch import optim
import numpy as np

def setup_optimizer(model, lr, weight_decay):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    # if lr_schedule is None:
    #     scheduler = None
    # else:
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * steps_per_epoch)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer#, scheduler

def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).to(diffusion_steps.device)
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


# Unused I think
# https://github.com/mikesha2/kolmogorov_smirnov_torch/blob/main/ks_test.py
"""
Created on Wed Mar 15 11:10:03 2023

@author: mikesha
"""
import torch

def alpha_D(D, n1: int, n2: int):
    return 2 * (-D.square() * 2 * n1 / (1 + n1 / n2)).exp()

@torch.jit.script
def kolmogorov_smirnov(points1, points2, alpha=torch.as_tensor([0.05, 0.01, 0.001, 0.0001])):
    """
    Kolmogorov-Smirnov test for empirical similarity of probability distributions.
    
    Warning: we assume that none of the elements of points1 coincide with points2. 
    The test may gave false negatives if there are coincidences, however the effect
    is small.

    Parameters
    ----------
    points1 : (..., n1) torch.Tensor
        Batched set of samples from the first distribution
    points2 : (..., n2) torch.Tensor
        Batched set of samples from the second distribution
    alpha : torch.Tensor
        Confidence intervals we wish to test. The default is torch.as_tensor([0.05, 0.01, 0.001, 0.0001]).

    Returns
    -------
    Tuple of (torch.Tensor, torch.Tensor)
        The test result at each alpha, and the estimated p-values.

    """
    n1 = points1.shape[-1]
    n2 = points2.shape[-1]
    
    # Confidence level
    c_ks = torch.sqrt(-0.5 * (alpha / 2).log())
    sup_conf = c_ks * torch.as_tensor((n1 + n2) / (n1 * n2)).sqrt()
    sup_conf = sup_conf.reshape((1, alpha.shape[0]))
    
    comb = torch.concatenate((points1, points2), dim=-1)
    
    comb_argsort = comb.argsort(dim=-1)
    
    pdf1 = torch.where(comb_argsort < n1, 1 / n1, 0)
    pdf2 = torch.where(comb_argsort >= n1, 1 / n2, 0)
    
    cdf1 = pdf1.cumsum(dim=-1)
    cdf2 = pdf2.cumsum(dim=-1)
    
    sup, _ = (cdf1 - cdf2).abs().max(dim=-1, keepdim=True)
    return sup > sup_conf, alpha_D(sup, n1 ,n2)