from agent.config import *
from agent.expression import Expression
from agent.tokens import *
import torch
import numpy as np
from typing import Dict, Any

def parse_token(token_str: str) -> Token:
    token_str = token_str.lower()

    for feature in FeatureType:
        if token_str == feature.name.lower():
            return FeatureToken(feature)
        
    for op in OPERATORS:
        if token_str == op.__name__.lower():
            return OperatorToken(op)
        
    try:
        val = float(token_str)
        return ConstantToken(val)
    except ValueError:
        pass

    if token_str.startswith('dt'):
        try:
            delta = int(token_str[2:])
            return DeltaTimeToken(delta)
        except ValueError:
            pass

    if token_str == 'sep':
        return SEP_TOKEN
    
    raise ValueError(f"Unknown token: {token_str}")
  
def compute_ic_series(
        f_all: torch.Tensor,
        ret_all: torch.Tensor,
        min_obs: int=500
):
    
    T, N = f_all.shape
    ic_list = torch.zeros(T, device=f_all.device, dtype=torch.float32)

    for t in range(T):
        f = f_all[t]
        ret = ret_all[t]
        mask = ~torch.isnan(f) & ~torch.isnan(ret)
        if mask.sum() < min_obs:
            ic_list[t] = 0.0
            continue
        f_sel = f[mask]
        ret_sel = ret[mask]
        f_mean = f_sel.mean()
        ret_mean = ret_sel.mean()
        cov = ((f_sel - f_mean) * (ret_sel - ret_mean)).sum()
        std_f = torch.std(f_sel)
        std_ret = torch.std(ret_sel)
        denom = std_f * std_ret
        if denom.item() == 0:
            ic_list[t] = 0.0
        else:
            ic_list[t] = (cov / denom).item()

    return ic_list

