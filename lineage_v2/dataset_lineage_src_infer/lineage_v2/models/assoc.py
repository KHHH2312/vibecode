"""Sparse parental association head (Stage B skeleton)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParentalAssocHead(nn.Module):
    """Score (child, candidate_parent) pairs + quiet logit.

    Features: child emb, parent emb, delta_um (3), |delta|, flow residual (3), dist.
    """

    def __init__(self, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        in_dim = emb_dim * 2 + 3 + 1 + 3 + 1  # emb_c, emb_p, dzyx, dist, flow_res, flow_norm
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.quiet_mlp = nn.Sequential(
            nn.Linear(emb_dim + 4, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def edge_logit(
        self,
        emb_c: torch.Tensor,
        emb_p: torch.Tensor,
        child_um: torch.Tensor,
        parent_um: torch.Tensor,
        child_flow_um: torch.Tensor,
    ) -> torch.Tensor:
        # emb_*: (E, D), coords (E, 3)
        delta = parent_um - child_um
        dist = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        parent_hat = child_um + child_flow_um
        flow_res = parent_um - parent_hat
        flow_n = flow_res.norm(dim=-1, keepdim=True)
        x = torch.cat([emb_c, emb_p, delta, dist, flow_res, flow_n], dim=-1)
        return self.edge_mlp(x).squeeze(-1)

    def quiet_logit(self, emb_c: torch.Tensor, child_flow_um: torch.Tensor) -> torch.Tensor:
        fn = child_flow_um.norm(dim=-1, keepdim=True)
        x = torch.cat([emb_c, child_flow_um, fn], dim=-1)
        return self.quiet_mlp(x).squeeze(-1)
