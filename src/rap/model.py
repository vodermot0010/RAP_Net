from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import MLP, PolylineEncoder, StateDropout, AttentionBlock, SelfAttentionBlock


@dataclass
class RAPOutput:
    traj_proposal: torch.Tensor   # (B, N, K, T, 4)
    traj_refined: torch.Tensor    # (B, N, K, T, 4)
    mode_logits: torch.Tensor     # (B, K)


class RAPLITE(nn.Module):
    '''
    RAP-Lite model:

    - Role-aware encoder:
        * identity tokens: T_AV, T_AGENT
        * route token: T_ROUTE added only to map elements on the ego route
      This is the "Route–Identity Token Pairing" mechanism.

    - Multimodal interaction decoder:
        K modes, each interaction layer:
          (1) Spatio-temporal cross-attention:
                - query -> map tokens (spatial context incl. route)
                - query -> own history tokens (temporal context)
          (2) Agent attention within each mode
          (3) Mode attention within each agent

    - Optional proposal refinement stage (proposal queries re-injected to decoder)

    Inputs (batch):
      agent_hist: (B, N, H, F_hist)
      agent_hist_valid: (B, N, H) bool
      agent_mask: (B, N) bool (True=valid)
      map_poly: (B, M, P, F_map)
      map_on_route: (B, M) bool
      map_mask: (B, M) bool (True=valid)
    '''
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        num_modes: int = 6,
        dropout: float = 0.1,
        state_dropout_prob: float = 0.0,
        fut_len: int = 16,
        use_refine: bool = True,
        hist_in_dim: int = 7,
        map_in_dim: int = 6,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_modes = num_modes
        self.fut_len = fut_len
        self.use_refine = use_refine

        # ---- tokens (learnable) ----
        self.t_av = nn.Parameter(torch.zeros(d_model))
        self.t_agent = nn.Parameter(torch.zeros(d_model))
        self.t_route = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.t_av, std=0.02)
        nn.init.normal_(self.t_agent, std=0.02)
        nn.init.normal_(self.t_route, std=0.02)

        # ---- encoders ----
        self.state_dropout = StateDropout(state_dropout_prob)
        self.hist_mlp = MLP(hist_in_dim, d_model, d_model, num_layers=3, dropout=dropout)
        self.hist_ln = nn.LayerNorm(d_model)

        self.map_encoder = PolylineEncoder(map_in_dim, d_model, hidden_dim=d_model, dropout=dropout)

        # ---- mode embeddings ----
        self.mode_emb = nn.Parameter(torch.randn(num_modes, d_model) * 0.02)

        # ---- decoder blocks ----
        self.map_attn = nn.ModuleList([AttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.hist_attn = nn.ModuleList([AttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.agent_attn = nn.ModuleList([SelfAttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.mode_attn = nn.ModuleList([SelfAttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)])

        # ---- output heads ----
        # separate heads for ego vs other agents (role-aware output)
        self.head_ego = MLP(d_model, d_model, fut_len * 4, num_layers=3, dropout=dropout)
        self.head_agent = MLP(d_model, d_model, fut_len * 4, num_layers=3, dropout=dropout)

        # mode scoring (joint future score)
        self.score_mlp = MLP(d_model, d_model, 1, num_layers=2, dropout=dropout)

        # refinement
        if use_refine:
            self.proposal_enc = MLP(fut_len * 4, d_model, d_model, num_layers=3, dropout=dropout)
            self.ref_map_attn = nn.ModuleList([AttentionBlock(d_model, nhead, dropout) for _ in range(1)])
            self.ref_hist_attn = nn.ModuleList([AttentionBlock(d_model, nhead, dropout) for _ in range(1)])
            self.ref_agent_attn = nn.ModuleList([SelfAttentionBlock(d_model, nhead, dropout) for _ in range(1)])
            self.ref_mode_attn = nn.ModuleList([SelfAttentionBlock(d_model, nhead, dropout) for _ in range(1)])
            self.ref_head_ego = MLP(d_model, d_model, fut_len * 4, num_layers=3, dropout=dropout)
            self.ref_head_agent = MLP(d_model, d_model, fut_len * 4, num_layers=3, dropout=dropout)

    def encode_history(self, agent_hist: torch.Tensor, agent_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        agent_hist: (B, N, H, F_hist)
        agent_mask: (B, N) bool
        returns:
          hist_tokens: (B, N, H, D)
          cur_tokens:  (B, N, D) last-timestep embedding + identity token
        '''
        B, N, H, Fdim = agent_hist.shape
        x = self.state_dropout(agent_hist)
        x = self.hist_mlp(x)  # (B,N,H,D)
        x = self.hist_ln(x)

        # identity tokens
        id_tok = torch.zeros((B, N, 1, self.d_model), device=x.device, dtype=x.dtype)
        id_tok[:, 0, 0] = self.t_av
        if N > 1:
            id_tok[:, 1:, 0] = self.t_agent
        x = x + id_tok  # broadcast over time

        # mask invalid agents (pad agents)
        x = x * agent_mask[:, :, None, None].float()

        cur = x[:, :, -1]  # (B,N,D)
        return x, cur

    def encode_map(self, map_poly: torch.Tensor, map_on_route: torch.Tensor, map_mask: torch.Tensor) -> torch.Tensor:
        '''
        map_poly: (B,M,P,F_map)
        map_on_route: (B,M) bool
        map_mask: (B,M) bool
        returns:
          map_tokens: (B,M,D) + route token for route lanes
        '''
        m = self.map_encoder(map_poly, mask=map_mask)  # (B,M,D)
        # add route token only to on-route lanes (route-aware conditioning)
        m = m + map_on_route.unsqueeze(-1).float() * self.t_route
        return m

    def _decode_once(
        self,
        queries: torch.Tensor,             # (B,N,K,D)
        hist_tokens: torch.Tensor,         # (B,N,H,D)
        map_tokens: torch.Tensor,          # (B,M,D)
        agent_mask: torch.Tensor,          # (B,N) True=valid
        map_mask: torch.Tensor,            # (B,M) True=valid
        layers: int,
        map_attn_blocks,
        hist_attn_blocks,
        agent_attn_blocks,
        mode_attn_blocks,
    ) -> torch.Tensor:
        B, N, K, D = queries.shape
        # Cross-attn to map (flatten queries as (B, NK, D))
        q_flat = queries.reshape(B, N * K, D)
        # PyTorch key_padding_mask uses True for PAD
        map_kpm = (~map_mask).bool()  # (B,M)

        # Cross-attn to history (per-agent, so treat BN as batch)
        hist_bn = hist_tokens.reshape(B * N, hist_tokens.shape[2], D)  # (BN,H,D)
        agent_kpm = (~agent_mask).bool()  # (B,N) True=pad

        for l in range(layers):
            # (1) map cross-attention
            q_map = map_attn_blocks[l](q_flat, map_tokens, map_tokens, key_padding_mask=map_kpm)  # (B,NK,D)

            # (2) history cross-attention (own history)
            q_per_agent = q_flat.reshape(B, N, K, D).permute(0, 1, 2, 3).reshape(B * N, K, D)
            # history key padding: (BN,H) from agent_hist_valid could be used; we assume all valid in toy.
            q_hist = hist_attn_blocks[l](q_per_agent, hist_bn, hist_bn, key_padding_mask=None)  # (BN,K,D)
            q_hist = q_hist.reshape(B, N, K, D).reshape(B, N * K, D)

            # sum (spatio-temporal)
            q_flat = q_map + q_hist

            # (3) agent attention within each mode
            q_mode = q_flat.reshape(B, N, K, D).permute(0, 2, 1, 3)  # (B,K,N,D)
            q_mode_flat = q_mode.reshape(B * K, N, D)
            agent_kpm_rep = agent_kpm.unsqueeze(1).expand(B, K, N).reshape(B * K, N)
            q_mode_flat = agent_attn_blocks[l](q_mode_flat, key_padding_mask=agent_kpm_rep)
            q_mode = q_mode_flat.reshape(B, K, N, D).permute(0, 2, 1, 3)  # (B,N,K,D)

            # (4) mode attention across modes per agent
            q_agent = q_mode.permute(0, 1, 2, 3)  # (B,N,K,D)
            q_agent_flat = q_agent.reshape(B * N, K, D)
            # no padding across modes
            q_agent_flat = mode_attn_blocks[l](q_agent_flat, key_padding_mask=None)
            q_agent = q_agent_flat.reshape(B, N, K, D)

            q_flat = q_agent.reshape(B, N * K, D)

        return q_flat.reshape(B, N, K, D)

    def forward(self, batch: Dict[str, torch.Tensor]) -> RAPOutput:
        agent_hist = batch["agent_hist"]          # (B,N,H,F)
        agent_mask = batch["agent_mask"]          # (B,N)
        map_poly = batch["map_poly"]              # (B,M,P,Fm)
        map_on_route = batch["map_on_route"]      # (B,M)
        map_mask = batch["map_mask"]              # (B,M)

        B, N, H, Fdim = agent_hist.shape
        T = self.fut_len
        K = self.num_modes

        hist_tokens, cur_tokens = self.encode_history(agent_hist, agent_mask)   # (B,N,H,D), (B,N,D)
        map_tokens = self.encode_map(map_poly, map_on_route, map_mask)          # (B,M,D)

        # init mode queries: q_{i,k} = cur_token_i + mode_emb_k
        q = cur_tokens[:, :, None, :] + self.mode_emb[None, None, :, :]  # (B,N,K,D)

        # decode
        q = self._decode_once(
            q, hist_tokens, map_tokens, agent_mask, map_mask,
            layers=self.num_layers,
            map_attn_blocks=self.map_attn,
            hist_attn_blocks=self.hist_attn,
            agent_attn_blocks=self.agent_attn,
            mode_attn_blocks=self.mode_attn,
        )  # (B,N,K,D)

        # output proposals
        # role-aware heads: ego uses head_ego; other agents use head_agent
        q_ego = q[:, 0]                # (B,K,D)
        q_agent = q[:, 1:] if N > 1 else None

        traj_ego = self.head_ego(q_ego).reshape(B, K, T, 4)
        if q_agent is not None:
            traj_ag = self.head_agent(q_agent).reshape(B, N - 1, K, T, 4)
            traj_prop = torch.cat([traj_ego[:, None, ...], traj_ag], dim=1)  # (B,N,K,T,4)
        else:
            traj_prop = traj_ego[:, None, ...]  # (B,1,K,T,4)

        # mode logits: max pool across agent dimension on q
        q_pool = q.max(dim=1).values  # (B,K,D)
        mode_logits = self.score_mlp(q_pool).squeeze(-1)  # (B,K)

        traj_ref = traj_prop
        if self.use_refine:
            # proposal-based refinement: encode trajectory proposal as query, run 1-layer refinement
            prop_feat = traj_prop.reshape(B, N, K, T * 4)
            q2 = self.proposal_enc(prop_feat)  # (B,N,K,D)
            q2 = self._decode_once(
                q2, hist_tokens, map_tokens, agent_mask, map_mask,
                layers=1,
                map_attn_blocks=self.ref_map_attn,
                hist_attn_blocks=self.ref_hist_attn,
                agent_attn_blocks=self.ref_agent_attn,
                mode_attn_blocks=self.ref_mode_attn,
            )
            q2_ego = q2[:, 0]
            q2_agent = q2[:, 1:] if N > 1 else None
            dtraj_ego = self.ref_head_ego(q2_ego).reshape(B, K, T, 4)
            if q2_agent is not None:
                dtraj_ag = self.ref_head_agent(q2_agent).reshape(B, N - 1, K, T, 4)
                dtraj = torch.cat([dtraj_ego[:, None, ...], dtraj_ag], dim=1)
            else:
                dtraj = dtraj_ego[:, None, ...]
            traj_ref = traj_prop + dtraj

        return RAPOutput(traj_proposal=traj_prop, traj_refined=traj_ref, mode_logits=mode_logits)
