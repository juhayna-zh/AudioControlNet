from typing import Optional

import torch


def log_normal_sample(x: torch.Tensor,
                      generator: Optional[torch.Generator] = None,
                      m: float = 0.0,
                      s: float = 1.0) -> torch.Tensor:
    bs = x.shape[0]
    s = torch.randn(bs, device=x.device, generator=generator) * s + m
    return torch.sigmoid(s)
import torch
from typing import Optional, Tuple

def log_normal_sample_r_t(
    x: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    m: float = 0.0,
    s: float = 1.0,
    epsilon: float = 1.0  
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate two tensors, ensuring that every element of the second tensor is greater than the corresponding element of the first tensor.

    Parameters:
    x (torch.Tensor): Input tensor (used to determine batch size and device)
    generator (torch.Generator, optional): Random number generator
    m (float): Mean of the normal distribution (default: 0)
    s (float): Standard deviation of the normal distribution (default: 1)
    epsilon (float): Controls the minimum increment of the second tensor (default: 1)

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: Two tensors after applying the sigmoid function, where every element of the second tensor is greater than the corresponding element of the first one
    """
    bs = x.shape[0]
    device = x.device
    
    s1 = torch.randn(bs, device=device, generator=generator) * s + m
    
    increment = torch.abs(torch.randn(bs, device=device, generator=generator)) * epsilon
    s2 = s1 + increment
    
    return torch.sigmoid(s1), torch.sigmoid(s2)