from re import M
from typing import Dict, List, OrderedDict

from agent.data import DataContainer
from agent.expr_tree import ExprTree
from agent.expression import Expression
from agent.utils import parse_token
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sympy import factor
from torch import Tensor
import torch

class AlphaPool:
    def __init__(self, rpn_exprs: List[List[str]], data: DataContainer):
        self.data          = data
        self.expressions   = []
        self.factor_values = []
        self.T = self.data['return'].shape[0]
        # 1) load initial expressions
        for rpn in rpn_exprs:
            self.add_expression(rpn)
        
        self._compute_baseline()

    def add_expression(self, rpn: List[str]):
        tokens = [parse_token(token) for token in rpn]
        tree = ExprTree()
        for token in tokens:
            tree.push(token)
        if tree.is_valid():
            expr = tree.get_stack()
            self.expressions.append(expr)
            values = expr.evaluate(self.data)
            if values.shape[0] < self.T:
                self.T = values.shape[0]
            self.factor_values.append(values)

    def _compute_baseline(self):
        """
        Precompute baseline R²_b for each time‐slice using statsmodels OLS.
        """
        # 1) slice out returns at your chosen indices
        self.t_idx = torch.arange(1, self.T, 20, device=self.data['return'].device)
        ret_all = (
            self.data['return'][:self.T]
                .index_select(0, self.t_idx)
                .cpu()
                .numpy()
        )  # shape (M, N)

        # 2) stack baseline factors into (M, N, k)
        if self.factor_values:
            Xb_all = torch.stack(
                [fv[:self.T].index_select(0, self.t_idx) for fv in self.factor_values],
                dim=2
            ).cpu().numpy()
        else:
            M, N = ret_all.shape
            Xb_all = np.empty((M, N, 0), dtype=float)

        M, N = ret_all.shape
        self.r2b_list: List[float] = []

        for m in range(M):
            y = ret_all[m]             # (N,)
            Xb = Xb_all[m]             # (N, k)

            # mask out any NaNs
            mask = np.isfinite(y) & np.all(np.isfinite(Xb), axis=1)
            if mask.sum() < 10:        # skip tiny cross‐sections
                self.r2b_list.append(0.0)
                continue

            y_m  = y[mask]
            Xb_m = Xb[mask]

            # build a DataFrame so we can call statsmodels conveniently
            df = pd.DataFrame(Xb_m)
            # OLS will automatically add a constant when we call sm.add_constant
            model = sm.OLS(y_m, sm.add_constant(df), missing='drop')
            res   = model.fit()
            self.r2b_list.append(res.rsquared)
    
    def compute_marginal_contribution(self, new_expr: Expression) -> float:
        # 1) slice returns and new factor
        ret_all = (
            self.data['return'][:self.T]
                .index_select(0, self.t_idx)
                .cpu()
                .numpy()
        )  # (M, N)
        f_all = (
            new_expr.evaluate(self.data)[:self.T]
                .index_select(0, self.t_idx)
                .cpu()
                .numpy()
        )  # (M, N)

        # 2) re-stack baseline factors into (M, N, k)
        if self.factor_values:
            Xb_all = torch.stack(
                [fv[:self.T].index_select(0, self.t_idx) for fv in self.factor_values],
                dim=2
            ).cpu().numpy()
        else:
            M, N = ret_all.shape
            Xb_all = np.empty((M, N, 0), dtype=float)

        M, N = ret_all.shape
        r2w_list: List[float] = []

        for m in range(M):
            y  = ret_all[m]            # (N,)
            Xb = Xb_all[m]             # (N, k)
            fn = f_all[m]              # (N,)

            # mask out NaNs in y, Xb or fn
            mask = np.isfinite(y) \
                   & np.all(np.isfinite(Xb), axis=1) \
                   & np.isfinite(fn)
            if mask.sum() < 500:
                continue

            y_m   = y[mask]
            Xb_m  = Xb[mask]
            fn_m  = fn[mask][:, None]   # (n,1)

            # build DataFrame [Xb_m | fn_m]
            df = pd.DataFrame(np.hstack([Xb_m, fn_m]))
            model = sm.OLS(y_m, sm.add_constant(df), missing='drop')
            res   = model.fit()
            r2w_list.append(res.rsquared)

        if not r2w_list:
            return 0.0

        # turn both lists into numpy
        r2b = np.array(self.r2b_list)
        r2w = np.array(r2w_list)

        # mean(R2_w) - mean(R2_b)
        return float(r2w.mean() - r2b.mean())
    
# def evaluate_metrics(
#         expr: Expression,
#         data: DataContainer,
#         long_pct: float = 0.3,
#         short_pct: float = 0.3,
#         min_obs: int = 500
# ) -> Dict[str, float]:
#     f_all = expr.evaluate(data)  # (T, N)
#     ret_all = data['return'][:f_all.shape[0]]  # (T, N)

#     ic_series = compute_ic_series(f_all, ret_all, min_obs)
#     pnl = compute_daily_pnl(f_all, ret_all, long_pct, short_pct, min_obs)

#     avg_ic = float(ic_series.nanmean().item())
#     avg_pnl = float(pnl.nanmean().item())
#     sharpe = compute_sharpe_ratio(pnl)
#     ir_ic = compute_ir(ic_series)

#     return OrderedDict([
#         ('avg_ic', avg_ic),
#         ('avg_pnl', avg_pnl),
#         ('sharpe', sharpe),
#         ('ir_ic', ir_ic),
#     ])

# class AlphaPool:
#     def __init__(self, rpn_exprs: List[List[str]], data: DataContainer):
#         self.data = data
#         self.expressions: List[Expression] = []
#         self.factor_values: List[torch.Tensor] = []
#         self.T = self.data['return'].shape[0]
#         for rpn in rpn_exprs:
#             self.add_expression(rpn)
    # def add_expression(self, rpn: List[str]):
    #     tokens = [parse_token(token) for token in rpn]
    #     tree = ExprTree()
    #     for token in tokens:
    #         tree.push(token)
    #     if tree.is_valid():
    #         expr = tree.get_stack()
    #         self.expressions.append(expr)
    #         values = expr.evaluate(self.data)
    #         if values.shape[0] < self.T:
    #             self.T = values.shape[0]
    #         self.factor_values.append(values)

#     def compute_marginal_contribution(self, new_expr: Expression) -> float:
#             ret = self.data['return']         
#             f = new_expr.evaluate(self.data)             
#             fs = self.factor_values                      
#             self.T = min(self.T, f.shape[0])
#             mc_list = []
#             N = ret.shape[1]
#             # select every 10th time step to reduce computation
#             t_idx = torch.arange(1, self.T, 20, device=ret.device)
#             M = t_idx.size(0)

#             y_all = ret.index_select(0, t_idx) # (M, N)
#             f_all = f.index_select(0, t_idx) # (M, N)
#             Xb_all = torch.stack([fs_j.index_select(0, t_idx) for fs_j in fs], dim=-1) if fs else torch.empty(M, N, 0, device=ret.device) # (M, N, k)
            
#             valid = ~torch.isnan(y_all) & ~torch.isnan(f_all)
#             if Xb_all.shape[2] > 0:
#                 valid &= ~torch.isnan(Xb_all).any(dim=2)

#             ones_all = torch.ones((M,N,1), device=ret.device)
#             Xb_design = torch.cat([ones_all, Xb_all], dim=2)                 # (M, N, k+1)
#             Xw_design = torch.cat([ones_all, Xb_all, f_all.unsqueeze(2)], dim=2)  # (M, N, k+2)
#             # for t in range(1, self.T, 10):
#             for m in range(M):
#                 # y = ret[t]         # (N,)
#                 # fi = f[t]          # (N,)
#                 # Xb = torch.stack([fs_j[t] for fs_j in fs], dim=1) if fs else torch.empty(N, 0, device=y.device)

#                 # mask = ~torch.isnan(y) & ~torch.isnan(fi)
#                 # if Xb.shape[1] > 0:
#                 #     mask &= ~torch.isnan(Xb).any(dim=1)
#                 # y_masked = y[mask]
#                 # fi_masked = fi[mask].unsqueeze(1)  # (m, 1)
#                 # Xb_masked = Xb[mask]               # (m, k)
#                 mask = valid[m]

#                 # if y_masked.numel() == 0:
#                 if not mask.any():
#                     continue

#                 # ones = torch.ones_like(y_masked).unsqueeze(1)
#                 # Xb_full = torch.cat([ones, Xb_masked], dim=1)                 # (m, k+1)
#                 # Xw_full = torch.cat([ones, Xb_masked, fi_masked], dim=1)     # (m, k+2)
#                 y_masked = y_all[m][mask]
#                 Xb_full = Xb_design[m][mask]  # (m, k+1)
#                 Xw_full = Xw_design[m][mask]  # (m, k+2)

#                 try:
#                     beta_b, _, _, _ = torch.linalg.lstsq(Xb_full, y_masked.unsqueeze(1))
#                     beta_w, _, _, _ = torch.linalg.lstsq(Xw_full, y_masked.unsqueeze(1))
#                 except RuntimeError:
#                     continue  # numerical issue

#                 ybar = y_masked.mean()
#                 ss_tot = ((y_masked - ybar)**2).sum()
#                 if ss_tot <= 0:
#                     continue

#                 yhat_b = (Xb_full @ beta_b).squeeze()
#                 yhat_w = (Xw_full @ beta_w).squeeze()

#                 ss_reg_b = ((yhat_b - ybar)**2).sum()
#                 ss_reg_w = ((yhat_w - ybar)**2).sum()

#                 r2_b = ss_reg_b / ss_tot
#                 r2_w = ss_reg_w / ss_tot
#                 mc_list.append((r2_w - r2_b).item())

#             return float(np.nanmean(mc_list)) if mc_list else 0.0


    
        