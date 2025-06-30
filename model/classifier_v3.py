import torch
from torch import nn
from .diffusion_model import Wavenet
from .util import calc_diffusion_step_embedding
from einops.layers.torch import Rearrange
from einops import rearrange # LOL
import numpy as np
from math import sqrt

INIT_STD = 0.02

class MHAStack(nn.Module):
    def __init__(
        self,
        d_embed,
        d_kv_embed,
        num_heads,
        ff,
        struct="scf", # self attn, cross attn, ff
        dropout=0,
        d_adap=0,
        depth=0,
    ):
        super().__init__()
        self.struct = struct
        self.cross_count = struct.count("c")
        
        layers = []
        for s in struct:
            match s:
                case "s":
                    l = nn.MultiheadAttention(
                        d_embed,
                        num_heads,
                        kdim=d_embed,
                        vdim=d_embed,
                        batch_first=True,
                        dropout=dropout
                    )
                    nn.init.trunc_normal_(l.in_proj_weight, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
                    l.in_proj_weight.data.div_(sqrt(2.0 * (depth + 1)))
                case "c":
                    l = nn.MultiheadAttention(
                        d_embed,
                        num_heads,
                        kdim=d_kv_embed,
                        vdim=d_kv_embed,
                        batch_first=True,
                        dropout=dropout
                    )
                    if d_embed == d_kv_embed:
                        nn.init.trunc_normal_(l.in_proj_weight, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
                        l.in_proj_weight.data.div_(sqrt(2.0 * (depth + 1)))
                    else:
                        nn.init.trunc_normal_(l.q_proj_weight, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
                        nn.init.trunc_normal_(l.k_proj_weight, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
                        nn.init.trunc_normal_(l.v_proj_weight, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
                        l.q_proj_weight.data.div_(sqrt(2.0 * (depth + 1)))
                        l.k_proj_weight.data.div_(sqrt(2.0 * (depth + 1)))
                        l.v_proj_weight.data.div_(sqrt(2.0 * (depth + 1)))

                case "f":
                    l = nn.Sequential(
                        nn.Linear(d_embed, d_embed * ff),
                        nn.GELU(),
                        nn.Linear(d_embed * ff, d_embed),
                    )
                    nn.init.trunc_normal_(l[0].weight, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
                    nn.init.constant_(l[0].bias, 0)
                    nn.init.trunc_normal_(l[2].weight, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
                    l[2].weight.data.div_(sqrt(2.0 * (depth + 1)))
                    nn.init.constant_(l[2].bias, 0)
                    depth += 1

            layers.append(l)
        
        self.res_drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        self.norms = nn.ModuleList([nn.LayerNorm(d_embed) for _ in layers])
        for n in self.norms:
            nn.init.constant_(n.bias, 0)
            nn.init.constant_(n.weight, 1.0)
        
        self.layers = nn.ModuleList(layers)

        self.have_adap = d_adap > 0
        if d_adap > 0:
            self.adaps = nn.ModuleList([nn.Linear(d_adap, 3 * d_embed) for _ in layers])
            for a in self.adaps:
                nn.init.zeros_(a.weight)
                nn.init.zeros_(a.bias)
                
    
    def forward(self, x, y=None, c=None):
        if y is not None:
            y = y.unbind(dim=1) # B Q C H -> Qx [B C H]
            assert len(y) == self.cross_count
        
        cross_idx = 0

        for idx, (s, n, l) in enumerate(zip(self.struct, self.norms, self.layers)):
            skip = x
            x = n(x)
            
            if self.have_adap and c is not None:
                adap_shift, adap_scale, adap_gate = self.adaps[idx](c).chunk(3, dim=-1)
                x = x * (1 + adap_scale) + adap_shift
            
            match s:
                case "s":
                    x, _ = l(x, x, x, need_weights=False)
                case "c":
                    x, _ = l(x, y[cross_idx], y[cross_idx], need_weights=False)
                    cross_idx += 1
                case "f":
                    x = l(x)
            
            if self.have_adap and c is not None:
                x = x * adap_gate
            
            x = self.res_drop(x)
            x = x + skip
        
        return x

class Classifier(nn.Module):
    def __init__(
        self,
        model: Wavenet,
        start=0,
        end=None,
        diffusion_t=1,

        query=["inter"], # inter, gate, filter
        reduce=["mean"], # mean, std
        rescale=False,
        L=1000,
        window_size=200,
        window_step=200,
        pool_merge="share", # mix, cat, share
        across_pool_struct=None,

        multi_query_merge="seq", # cat, seq, ind

        d_embed=None,
        num_heads=8,
        ff=4,
        stack_struct="cf",
        
        n_clst=1,

        final_stack_struct:str=None,
        n_class=6,
        final_act="pool", # cat, pool, cls
        
        dropout=0,

        have_ch_pos_embed=False,
        final_stack_pos_embed_dim=None,
    ):
        super().__init__()

        self.model = model
        for p in self.model.parameters():
            p.detach_()

        self.start = start
        self.end = end or len(model.layers)
        assert self.start < self.end and self.end <= len(model.layers)
        self.init_rearr = Rearrange("B C ... -> (B C) ...")

        self.query = query
        for q in query: assert q in ("inter", "gate", "filter")
        
        n_q = len(query)
        assert n_q == len(reduce)
        self.reduce = []
        self.rescale = []
        for r, q in zip(reduce, query):
            match q:
                case "gate":
                    w = 2
                    b = -1
                case _:
                    w = 1
                    b = 0
            match r:
                case "mean":
                    self.reduce.append(torch.mean)
                case "std":
                    self.reduce.append(torch.std)
                    w *= 2
                    b = -1
                case _: raise NotImplementedError(r)
            if rescale:
                self.rescale.append(lambda t: t * w + b)
            else:
                self.rescale.append(lambda t: t)
        # input
        # init rear
        # B C ... -> (B C) [1]...
        
        # query
        # (B C) ... -> ... -> (B C) n q L H
        
        # pool reduce
        # (B C) n q (p l) H -> (B C) n q p H l 
        # reduce -> ... -> (B C) n q p H
        
        # pool merge
        # mix ->   B n 1 q (p C)   H
        # cat ->   B n 1 q   C   (p H)
        # share -> B n p q   C     H
        #          B n P q   C     H

        # multiquery merge
        # cat -> B n 1 P 1 C (q H)
        # seq -> B n 1 P q C   H
        # ind -> B n q P 1 C   H
        #        B n T P Q C   H
        
        #             B  T  P
        clst_shape = [1, 1, 1, n_clst, d_embed]
        d_kv_embed = model.d_model
        n_tower = 1

        self.C = model.n_class
        if model.have_null_class: self.C -= 1
        self.cond = nn.Parameter(torch.arange(self.C, dtype=torch.long).unsqueeze(-1), requires_grad=False)
        self.diffusion_steps = nn.Parameter(torch.full((self.C, 1), diffusion_t, dtype=torch.long), requires_grad=False)
        
        assert (L - window_size) % window_step == 0
        self.L = L
        self.window_size = window_size
        self.window_step = window_step
        n_pool = (L - window_size) // window_step + 1
        
        self.pre_pool_rearr = Rearrange("B n q (p l) H -> B n q p H l", p=n_pool)

        match pool_merge:
            case "mix":
                self.pool_merge = Rearrange("(B C) n q p H -> B n 1 q (p C) H", C=self.C)
                # self.post_pool_merge = lambda x: (x,)
            case "cat":
                d_kv_embed *= n_pool
                self.pool_merge = Rearrange("(B C) n q p H -> B n 1 q C (p H)", C=self.C)
                # self.post_pool_merge = lambda x: (x,)
            case "share":
                self.pool_merge = Rearrange("(B C) n q p H -> B n p q C H", C=self.C)
                # self.post_pool_merge = lambda x: torch.unbind(x, dim=0)
            case _: raise NotImplementedError()

        self.q = n_q
        if n_q > 0:
            match multi_query_merge:
                case "cat":
                    d_kv_embed *= n_q
                    self.multi_query_merge = Rearrange("B n P q C H -> B n 1 P 1 C (q H)")
                    # self.post_multi_query_merge = lambda x: (x,)
                case "seq":
                    self.multi_query_merge = Rearrange("B n P q C H -> B n 1 P q C H")
                    # self.post_multi_query_merge = lambda x: (x,)
                case "ind":
                    n_tower = n_q
                    self.multi_query_merge = Rearrange("B n P q C H -> B n q P 1 C H")
                    # self.post_multi_query_merge = lambda x: x.chunk(self.q, dim=0)
                case _: raise NotImplementedError()
        else:
            self.multi_query_merge = Rearrange("B n P 1 C H -> B n 1 P 1 C H")
            # self.post_multi_query_merge = lambda x: x
        
        if have_ch_pos_embed:
            self.ch_pos_embed = nn.Parameter(
                calc_diffusion_step_embedding(
                    torch.arange(self.C).reshape(self.C, 1),
                    self.model.d_model
                # B n T P Q C H
                ).reshape(1, 1, 1, 1, 1, self.C, -1),
                requires_grad=False
            )
        else:
            self.ch_pos_embed = 0

        if d_embed is None:
            d_embed = d_kv_embed
            clst_shape[-1] = d_embed
        
        assert stack_struct != "" and stack_struct.count("c") == n_q
        cross_only = stack_struct.count("s") == 0
        assert cross_only or n_clst > 1
        

        self.cls_token = nn.Parameter(torch.zeros(clst_shape))
        nn.init.trunc_normal_(self.cls_token, std=INIT_STD, a=-INIT_STD, b=INIT_STD)

        self.t_pos_embed = nn.Parameter(torch.zeros(1, n_tower, 1, 1, 1))
        self.p_pos_embed = nn.Parameter(torch.zeros(1, 1, n_pool, 1, 1))
        nn.init.trunc_normal_(self.t_pos_embed, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
        nn.init.trunc_normal_(self.p_pos_embed, std=INIT_STD, a=-INIT_STD, b=INIT_STD)

        self.tower = nn.ModuleList([
            nn.ModuleList([
                MHAStack(
                    d_embed=d_embed,
                    d_kv_embed=d_kv_embed,
                    num_heads=num_heads,
                    ff=ff,
                    struct=stack_struct,
                    dropout=dropout,
                    d_adap=0,
                    depth=-0.5
                ) for __ in range(self.end - self.start)
            ]) for ___ in range(n_tower)
        ])
        
        assert across_pool_struct is None or across_pool_struct.count("c") == 0
        self.across_pool_tower = nn.ModuleList([
            nn.ModuleList([
                MHAStack(
                    d_embed=d_embed,
                    d_kv_embed=d_kv_embed,
                    num_heads=num_heads,
                    ff=ff,
                    struct=across_pool_struct,
                    dropout=dropout,
                    d_adap=0,
                    depth=-0.5
                ) for __ in range(self.end - self.start)
            ]) for ___ in range(n_tower)
        ]) if across_pool_struct is not None else [
            [
                None  for __ in range(self.end - self.start)
            ] for ___ in range(n_tower)
        ]

        if final_stack_struct is not None:
            assert final_stack_struct != "" and final_stack_struct.count("c") == 0
            if final_stack_pos_embed_dim is None: final_stack_pos_embed_dim = ""
            final_stack_pos_embed_shape = [1, n_tower, n_pool, n_clst, d_embed]
            if "T" not in final_stack_pos_embed_dim: final_stack_pos_embed_shape[1] = 1
            if "P" not in final_stack_pos_embed_dim: final_stack_pos_embed_shape[2] = 1
            if "N" not in final_stack_pos_embed_dim: final_stack_pos_embed_shape[3] = 1
            if np.prod(final_stack_pos_embed_shape) != d_embed:
                self.final_stack_pos_embed = nn.Parameter(torch.zeros(*final_stack_pos_embed_shape))
                nn.init.trunc_normal_(self.final_stack_pos_embed, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
            else:
                self.final_stack_pos_embed = 0
            
            self.final_stack = MHAStack(
                d_embed=d_embed,
                d_kv_embed=d_embed,
                num_heads=num_heads,
                ff=ff,
                struct=final_stack_struct,
                dropout=dropout,
                d_adap=0,
                depth=-0.5
            )
        else:
            self.final_stack_pos_embed = 0
            self.final_stack = lambda x: x

        match final_act:
            case "cat":
                self.final_act = lambda x: x.flatten(start_dim=1)
                self.linear = nn.Linear(np.prod(clst_shape).item(), n_class)
            case "pool":
                self.final_act = lambda x: x.mean(dim=1)
                self.linear = nn.Linear(d_embed, n_class)
            case "cls":
                assert (not cross_only) or final_stack_struct is not None
                self.final_act = lambda x: x[:, 0]
                self.linear = nn.Linear(d_embed, n_class)
            case _: raise NotImplementedError()
        
        nn.init.trunc_normal_(self.linear.weight, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
        nn.init.constant_(self.linear.bias, 0)
        
    def forward(self, input, data_is_cached=False):
        if not data_is_cached:
            all_tokens = self.compute_tokens(input)
        else:
            all_tokens = input

        B = all_tokens.shape[0]
        x = self.cls_token.expand(B, -1, -1, -1, -1) + self.t_pos_embed + self.p_pos_embed
        # B T P N H
        total_P = x.shape[2]
        
        x = x.unbind(dim=1) # [B P N H] * T
        x = [list(_x.unbind(dim=1)) for _x in x] #[ [B N H] * P] * T


        # B n T P Q C H
        # TODO most x's are not affecting each other some some sort of parallelism can be pulled off?
        # the only loop that cant be paralleled is the one eliminating n, because clst need to be passed to the next layer
        for T, (Bn_PQCH, tower, ac_tower) in enumerate(zip(all_tokens.unbind(dim=2), self.tower, self.across_pool_tower)): # eliminate T
            for B__PQCH, layer, ac_layer in zip(Bn_PQCH.unbind(dim=1), tower, ac_tower): # eliminate n
                for P, B___QCH in enumerate(B__PQCH.unbind(dim=1)): # eliminate P
                    x[T][P] = layer(x[T][P], B___QCH) # B N H each
                if ac_layer is not None:
                    ac_x = torch.cat(x[T], dim=1) # B (N * P) H
                    ac_x = ac_layer(ac_x)
                    x[T] = list(torch.chunk(ac_x, chunks=total_P, dim=1)) # [B N H] * P

        x = [torch.stack(_x, dim=1) for _x in x] # [B P N H] * T
        x = torch.stack(x, dim=1) # B T P N H
        x = x + self.final_stack_pos_embed

        x = rearrange(x, "B T P N H -> B (T P N) H")
        x = self.final_stack(x)
        x = self.final_act(x)
        x = self.linear(x)
        
        return x
    
    @torch.no_grad()
    def compute_tokens(self, input):
        x = input[0]
        local_cond = input[1] if len(input) > 1 else None

        B = x.shape[0]

        x = self.init_rearr(x)
        if x.dim() < 3: x = x.unsqueeze(1)
        if local_cond is not None:
            local_cond = self.init_rearr(local_cond)
            if local_cond.dim() < 4: local_cond = local_cond.unsqueeze(1)
        
        diffusion_steps = self.diffusion_steps.repeat(B, 1)
        cond = self.cond.repeat(B, 1)
        
        cond = self.model.calc_cond(diffusion_steps, cond)

        x = rearrange(x, "B H L -> B L H")
        x = self.model.in_layer(x)

        all_tokens = []
        for i, l in enumerate(self.model.layers):
            if i == self.end:
                break
            x, _, query_result = l(x, cond, local_cond, query=self.query, skip_skip=True)
            #     [(B C) (p l) H] x q
            if i >= self.start:
                tokens = torch.stack(query_result, dim=1) # (B C) q (p l) H
                all_tokens.append(tokens)

        all_tokens = torch.stack(all_tokens, dim=1)
        # (B C) n q (p l) H
        #   0   1 2   3   4

        assert all_tokens.shape[3] == self.L
        all_tokens = all_tokens.unfold(dimension=3, size=self.window_size, step=self.window_step)
        # (B C) n q p H l
        #   0   1 2 3 4 5

        # pool, reduce
        all_tokens = all_tokens.unbind(dim=2)
        # [(B C) n p H l] x q
        #    0   1 2 3 4
        
        temp = []
        for at, r, rs in zip(all_tokens, self.reduce, self.rescale):
            at = rs(r(at, dim=-1))
            temp.append(at)
        all_tokens = torch.stack(temp, dim=2) # (B C) n q p H
        # (B C) n q p H
        #   0   1 2 3 4
        
        all_tokens = self.pool_merge(all_tokens)
        # B n P q C H
        # 0 1 2 3 4 5


        all_tokens = self.multi_query_merge(all_tokens)
        # B n T P Q C H
        # 0 1 2 3 4 5 6

        return all_tokens + self.ch_pos_embed
