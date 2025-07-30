import torch
from torch import nn
from .diffusion_model import Wavenet
from .util import calc_diffusion_step_embedding
from einops.layers.torch import Rearrange
from einops import rearrange # LOL
import numpy as np
from math import sqrt
from functools import partial

INIT_STD = 0.02
param_init_fn = torch.rand

class LatentActivityExtractor(nn.Module):
    def __init__(
        self,
        model: Wavenet,
        start=0,
        end=None,
        diffusion_t=1,

        query=["inter"], # inter, gate, filter
    ):
        super().__init__()

        self.model = model
        for p in self.model.parameters():
            p.detach_()

        self.start = start
        self.end = end or len(model.layers)
        assert self.start < self.end and self.end <= len(model.layers)
        self.n_layer = self.end - self.start
        self.init_rearr = Rearrange("B C ... -> (B C) ...")

        self.query = query
        for q in query: assert q in ("inter", "gate", "filter")
        
        self.C = model.n_class
        if model.have_null_class: self.C -= 1
        self.cond = nn.Buffer(torch.arange(self.C, dtype=torch.long).unsqueeze(-1))
        self.diffusion_steps = nn.Buffer(torch.full((self.C, 1), diffusion_t, dtype=torch.long))

    @torch.no_grad()
    def forward(self, input, is_caching=None, rate=1):
        if False and is_caching:
            query = ["inter", "gate", "filter"]
            start = 0
            end = len(self.model.layers)
        else:
            query = self.query
            start = self.start
            end = self.end

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

        latent_activities = []
        for i, l in enumerate(self.model.layers):
            if i == end:
                break
            x, _, query_result = l(x, cond, local_cond, query=query, skip_skip=True, rate=rate)
            #     [(B C) (p l) H] x q
            if i >= start:
                tokens = torch.stack(query_result, dim=1) # (B C) q (p l) H
                latent_activities.append(tokens)

        latent_activities = torch.stack(latent_activities, dim=1)
        # (B C) n q (p l) H
        #   0   1 2   3   4
        latent_activities = rearrange(latent_activities, "(B C) ... -> B C ...", C = self.C)
        return latent_activities

    # def do_cache(self, input):
    #     return self(input, is_caching=True)

    # def from_cache(self, cached_input):
    #     idx = torch.tensor(
    #         [["inter", "gate", "filter"].index(q) for q in self.query],
    #         dtype=torch.long,
    #         device=cached_input.device
    #     )
    #     return cached_input[:, :, :, idx, :, :]

class LatentActivityReducer(nn.Module):
    def __init__(
        self,
        query=["inter"], # inter, gate, filter
        reduce=["mean"], # mean, std
        rescale=False, # TODO rescale can be computed after cache
        L=1000,
        window_size=200,
        window_step=200,
        pool_merge="share", # mix, cat, share
        multi_query_merge="seq", # cat, seq, ind
    ):
        super().__init__()
        # self.extractor = extractor
        # LatentActivityExtractor(
        #     model=model,
        #     start=start,
        #     end=end,
        #     diffusion_t=diffusion_t,
        #     query=query, # inter, gate, filter
        # )

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
                self.rescale.append(lambda x: x * w + b)
            else:
                self.rescale.append(lambda x: x)
        
        # input
        # init rear
        # B C ... -> (B C) [1]...
        
        # query
        # B C ... -> ... -> B C n q L H
        
        # pool reduce
        # B C n q (p l) H -> B C n q p H l 
        # reduce -> ... -> B C n q p H
        
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
        
        d_kv_embed_factor = 1
        
        assert (L - window_size) % window_step == 0
        self.L = L
        self.window_size = window_size
        self.window_step = window_step
        n_pool = (L - window_size) // window_step + 1
        self.n_pool = n_pool

        self.pre_pool_rearr = Rearrange("B C n q (p l) H -> B C n q p H l", p=n_pool)
        
        match pool_merge:
            case "mix":
                self.pool_merge = Rearrange("B C n q p H -> B n 1 q (p C) H")
            case "cat":
                d_kv_embed_factor *= n_pool
                self.pool_merge = Rearrange("B C n q p H -> B n 1 q C (p H)")
            case "share":
                self.pool_merge = Rearrange("B C n q p H -> B n p q C H")
            case _: raise NotImplementedError()

        self.n_query = n_q
        if n_q > 1:
            match multi_query_merge:
                case "cat":
                    d_kv_embed_factor *= n_q
                    self.multi_query_merge = Rearrange("B n P q C H -> B n 1 P 1 C (q H)")
                case "seq":
                    self.multi_query_merge = Rearrange("B n P q C H -> B n 1 P q C H")
                case "ind":
                    self.multi_query_merge = Rearrange("B n P q C H -> B n q P 1 C H")
                case _: raise NotImplementedError()
        else:
            self.multi_query_merge = Rearrange("B n P 1 C H -> B n 1 P 1 C H")
            self.multi_query_unpack = lambda x: x
            self.multi_query_repack = lambda x: x
        
        self.d_kv_embed_factor = d_kv_embed_factor

    @torch.no_grad()
    def forward(self, input, rate=1):
        if rate != 1: assert (_l := self.L * rate).is_integer() and input.shape[4] == _l
        # assert input.shape[4] == self.L
        all_tokens = input.unfold(dimension=4, size=self.window_size, step=self.window_step)
        # B C n q p H l
        # 0 1 2 3 4 5 6

        # pool, reduce
        all_tokens = all_tokens.unbind(dim=3)
        # [B C n p H l] x q
        #  0 1 2 3 4 5
        
        temp = []
        for at, r, rs in zip(all_tokens, self.reduce, self.rescale):
            at = rs(r(at, dim=-1))
            temp.append(at)
        all_tokens = torch.stack(temp, dim=3) # B C n q p H
        # B C n q p H
        # 0 1 2 3 4 5
        
        all_tokens = self.pool_merge(all_tokens)
        # B n P q C H
        # 0 1 2 3 4 5


        all_tokens = self.multi_query_merge(all_tokens)
        # B n T P Q C H
        # 0 1 2 3 4 5 6

        return all_tokens

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
        do_weight_init=False,
        have_crossnorm=True,
    ):
        super().__init__()
        self.struct = struct
        self.cross_count = struct.count("c")
        
        layers = []
        layers_by_depth = [[]]
        layer_depths = []
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
                case "c":
                    l = nn.MultiheadAttention(
                        d_embed,
                        num_heads,
                        kdim=d_kv_embed,
                        vdim=d_kv_embed,
                        batch_first=True,
                        dropout=dropout
                    )
                case "f":
                    l = nn.Sequential(
                        nn.Linear(d_embed, d_embed * ff),
                        nn.GELU(),
                        nn.Linear(d_embed * ff, d_embed),
                    )

            layers.append(l)
            layers_by_depth[-1].append(l)
            layer_depths.append(len(layers_by_depth) + depth)
            if s == "f": layers_by_depth.append([])
        if len(layers_by_depth[-1]) == 0: del layers_by_depth[-1]
        self.layers_by_depth = layers_by_depth # no need to be ModuleList
        self.layer_depths = layer_depths

        self.res_drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        self.norms = nn.ModuleList([nn.LayerNorm(d_embed) for _ in layers])
        self.have_crossnorm = have_crossnorm
        if have_crossnorm: self.c_norms = nn.ModuleList([nn.LayerNorm(d_kv_embed) for _ in range(self.cross_count)])
        else: self.c_norms = nn.ModuleList([nn.Identity() for _ in range(self.cross_count)])

        self.layers = nn.ModuleList(layers)

        self.have_adap = d_adap > 0
        if d_adap > 0:
            self.adaps = nn.ModuleList([nn.Linear(d_adap, 3 * d_embed) for _ in layers])
            # Necessary weight initialization for stability
            for a in self.adaps:
                nn.init.zeros_(a.weight)
                nn.init.zeros_(a.bias)
        
        if do_weight_init:
            self._init_weight(depth, d_embed == d_kv_embed)

        self.cumulative_depth = depth + self.struct.count("f")
    
    def _init_weight(self, depth, d_em_is_d_kv):
        for l, s in zip(self.layers, self.struct):
            match s:
                case "s":
                    nn.init.trunc_normal_(l.in_proj_weight, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
                    l.in_proj_weight.data.div_(sqrt(2.0 * (depth + 1)))
                case "c":
                    if d_em_is_d_kv:
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
                    nn.init.trunc_normal_(l[0].weight, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
                    nn.init.constant_(l[0].bias, 0)
                    nn.init.trunc_normal_(l[2].weight, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
                    l[2].weight.data.div_(sqrt(2.0 * (depth + 1)))
                    nn.init.constant_(l[2].bias, 0)
                    depth += 1
    
        for n in self.norms:
            nn.init.constant_(n.bias, 0)
            nn.init.constant_(n.weight, 1.0)
        
        if self.have_crossnorm:
            for n in self.c_norms:
                nn.init.constant_(n.bias, 0)
                nn.init.constant_(n.weight, 1.0)

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
                    _y = self.c_norms[cross_idx](y[cross_idx])
                    x, _ = l(x, _y, _y, need_weights=False)
                    cross_idx += 1
                case "f":
                    x = l(x)
            
            if self.have_adap and c is not None:
                x = x * adap_gate
            
            x = self.res_drop(x)
            x = x + skip
        
        return x

class LatentActivityDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_kv_embed_factor,
        n_layer,
        d_embed=None,
        
        init_weight=False,

        n_query=1,
        embed_query=False,
        d_query_embed=None,
        
        have_ch_pos_embed=False,
        cat_ch_pos_embed=False,
        ch_pos_emb_sym=None,
        ch_order=["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6", "T6-O2", "A1-T3", "T3-C3", "C3-CZ", "C4-CZ", "T4-C4", "A2-T4", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"],
        
        clst_dim="", # TP
        clst_pos_embed_dim="N", # TPN
        n_clst=4,
        
        n_pool=5,
        pool_merge="share",
        
        multi_query_merge="seq",
        
        stack_struct="scf",
        num_heads=8,
        ff=4,
        dropout=0,
        have_crossnorm=True,

        across_pool_stack_struct="",
        n_ap_clst=0,
        ap_clst_dim="T", # T
    ):
        super().__init__()
        d_kv_embed = d_model * d_kv_embed_factor

        self.embed_query = embed_query
        if embed_query:
            d_query_embed = d_query_embed or d_model
            self.query_embed_linear = nn.ModuleList([nn.Linear(d_model, d_query_embed) for _ in range(n_query)])
            d_kv_embed = d_query_embed
        
        self.have_ch_pos_embed = have_ch_pos_embed
        self.cat_ch_pos_embed = cat_ch_pos_embed
        if have_ch_pos_embed:
            match ch_pos_emb_sym:
                case "mirror":
                    self.ch_pos_embed_idx = nn.Buffer(torch.tensor(list(map(lambda c: (int(c.split("-")[0][-1]) % 2), ch_order)), dtype=torch.long),)
                    self.ch_pos_embed_raw = nn.Parameter(param_init_fn(1, 1, 1, 1, 1, 2, d_kv_embed))
                case None:
                    self.ch_pos_embed_idx = slice(None)
                    self.ch_pos_embed_raw = nn.Parameter(param_init_fn(1, 1, 1, 1, 1, len(ch_order), d_kv_embed))
                case _: raise NotImplementedError()
            if init_weight:nn.init.trunc_normal_(self.ch_pos_embed_raw, std=INIT_STD, a=-INIT_STD, b=INIT_STD)

            if cat_ch_pos_embed:
                d_kv_embed *= 2

        if d_embed is None: d_embed = d_kv_embed
        self.d_embed = d_embed
        
        match pool_merge:
            case "mix": n_pool = n_pool * len(ch_order)
            case "cat": n_pool = 1
            case "share": pass
            case _: raise NotImplementedError()
        if n_pool > 1 and "P" in clst_pos_embed_dim:
            self.p_pos_embed = nn.Parameter(param_init_fn(1, 1, n_pool, 1, d_embed))
            if init_weight: nn.init.trunc_normal_(self.p_pos_embed, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
        else:
            self.p_pos_embed = 0
        self.n_pool = n_pool

        n_tower = 1
        match multi_query_merge:
            case "cat":
                self.multi_query_unpack = partial(torch.chunk, chunks=n_query, dim=-1)
                self.multi_query_repack = partial(torch.cat, dim=-1)
            case "seq":
                self.multi_query_unpack = partial(torch.unbind, dim=-3)
                self.multi_query_repack = partial(torch.stack, dim=-3)
            case "ind":
                n_tower = n_query
                self.multi_query_unpack = partial(torch.unbind, dim=2)
                self.multi_query_repack = partial(torch.stack, dim=2)
            case _: raise NotImplementedError()
        if n_tower > 1:
            self.t_pos_embed = nn.Parameter(param_init_fn(1, n_tower, 1, 1, d_embed))
            if init_weight: nn.init.trunc_normal_(self.t_pos_embed, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
        else:
            self.t_pos_embed = 0
        self.n_tower = n_tower

        clst_shape = [1, 1, 1, n_clst, d_embed]
        if n_tower > 1 and "T" in clst_dim: clst_shape[1] = n_tower
        if n_pool > 1 and "P" in clst_dim: clst_shape[2] = n_pool

        self.cls_token = nn.Parameter(param_init_fn(clst_shape))
        if init_weight: nn.init.trunc_normal_(self.cls_token, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
        
        assert stack_struct.count("c") * n_tower == n_query
        cross_only = stack_struct.count("s") == 0
        assert cross_only or np.prod(clst_shape) > d_embed

        self.have_ap = across_pool_stack_struct is not None
        assert not self.have_ap or across_pool_stack_struct.count("c") == 0
        
        towers = []
        ap_towers = []
        for ___ in range(n_tower):
            layers = []
            ap_layers = []
            cumulative_depth = 0
            for __ in range(n_layer):
                layers.append(
                    MHAStack(
                        d_embed=d_embed,
                        d_kv_embed=d_kv_embed,
                        num_heads=num_heads,
                        ff=ff,
                        struct=stack_struct,
                        dropout=dropout,
                        d_adap=0,
                        depth=cumulative_depth,
                        do_weight_init=init_weight,
                        have_crossnorm=have_crossnorm,
                    )
                )
                cumulative_depth = layers[-1].cumulative_depth
                
                if self.have_ap:
                    ap_layers.append(
                        MHAStack(
                            d_embed=d_embed,
                            d_kv_embed=d_kv_embed,
                            num_heads=num_heads,
                            ff=ff,
                            struct=across_pool_stack_struct,
                            dropout=dropout,
                            d_adap=0,
                            depth=cumulative_depth,
                            do_weight_init=init_weight,
                            have_crossnorm=have_crossnorm,
                        )
                    )
                    cumulative_depth = ap_layers[-1].cumulative_depth
                            
            towers.append(nn.ModuleList(layers))
            ap_towers.append(nn.ModuleList(ap_layers))
        
        self.cumulative_depth = cumulative_depth
        self.tower = nn.ModuleList(towers)
        if self.have_ap: self.across_pool_tower = nn.ModuleList(ap_towers)
        else: self.across_pool_tower = [[None for _ in range(n_layer)] for _ in range(n_tower)]
        
        self.have_ap_clst = n_ap_clst > 0
        self.n_ap_clst = n_ap_clst
        if self.have_ap_clst:
            assert self.have_ap
            ap_clst_shape = [1, 1, n_ap_clst, d_embed]
            if "T" in ap_clst_dim: ap_clst_shape[1] = n_tower

            self.ap_cls_token = nn.Parameter(param_init_fn(ap_clst_shape))
            if init_weight: nn.init.trunc_normal_(self.ap_cls_token, std=INIT_STD, a=-INIT_STD, b=INIT_STD)

    def forward(self, all_tokens):
        if self.embed_query:
            all_tokens = self.multi_query_unpack(all_tokens)
            embedded_tokens = []
            for at, qel in zip(all_tokens, self.query_embed_linear):
                embedded_tokens.append(qel(at))
            all_tokens = self.multi_query_repack(embedded_tokens)
        
        if self.have_ch_pos_embed:
            pos_embed = self.ch_pos_embed_raw[..., self.ch_pos_embed_idx, :].expand(*all_tokens.shape[:-2], -1, -1)
            if self.cat_ch_pos_embed:
                all_tokens = torch.cat([all_tokens, pos_embed], dim=-1)
            else:
                all_tokens = all_tokens + pos_embed
        
        B = all_tokens.shape[0]
        x = self.cls_token.expand(B, self.n_tower, self.n_pool, -1, -1) + self.t_pos_embed + self.p_pos_embed
        # B T P N H
        
        x = x.unbind(dim=1) # [B P N H] * T
        x = [list(_x.unbind(dim=1)) for _x in x] #[ [B N H] * P] * T

        if self.have_ap_clst:
            ap_clst = self.ap_cls_token.expand(B, self.n_tower, -1, -1)
            ap_clst = ap_clst.unbind(dim=1) # [B A H] * T
        else:
            ap_clst = None

        # B n T P Q C H
        # TODO most x's are not affecting each other some some sort of parallelism can be pulled off?
        # the only loop that cant be paralleled is the one eliminating n, because clst need to be passed to the next layer
        for T, (Bn_PQCH, tower, ac_tower) in enumerate(zip(all_tokens.unbind(dim=2), self.tower, self.across_pool_tower)): # eliminate T
            for B__PQCH, layer, ap_layer in zip(Bn_PQCH.unbind(dim=1), tower, ac_tower): # eliminate n
                for P, B___QCH in enumerate(B__PQCH.unbind(dim=1)): # eliminate P
                    x[T][P] = layer(x[T][P], B___QCH) # B N H each
                
                if ap_layer is not None:
                    ap_x = x[T]
                    
                    if self.have_ap_clst:
                        ap_x = [ap_clst[T], *ap_x]
                    
                    ap_x = torch.cat(ap_x, dim=1) # B (A + N * P) H, 'A' may be 0
                    ap_x = ap_layer(ap_x)
                    
                    if self.have_ap_clst:
                        ap_clst[T] = ap_x[:, :self.n_ap_clst, :]
                        ap_x = ap_x[:, self.n_ap_clst:, :]
                    
                    x[T] = list(torch.chunk(ap_x, chunks=self.n_pool, dim=1)) # [B N H] * P

        if self.have_ap_clst: ap_clst = torch.stack(ap_clst, dim=1) # B T A H
        x = [torch.stack(_x, dim=1) for _x in x] # [B P N H] * T
        x = torch.stack(x, dim=1) # B T P N H

        return x, ap_clst

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        format="BTPNH",
        n_tower=0,
        n_pool=0,
        n_clst=0,
        n_ap_clst=0,
        d_embed=0,

        have_pos_embed=True,
        pos_embed_dim="TP", # TPN or TA
        stack_struct="sfsfsfsf",
        num_heads=8,
        ff=4,
        dropout=0,
        have_crossnorm=True,
        stack_init_depth=0,
        final_act="pool", # cat, pool, cls
        init_weight=False,
        n_class=6,
    ):
        super().__init__()
        match format:
            case "BTPNH":
                pos_embed_shape = [1, n_tower, n_pool, n_clst, d_embed]
            case "BTAH":
                pos_embed_shape = [1, n_tower, n_ap_clst, d_embed]
            case _: raise NotImplementedError()
        
        input_numel = np.prod(pos_embed_shape)
        for i, d in enumerate(format[:-1]):
            if d not in pos_embed_dim:
                pos_embed_shape[i] = 1
        if np.prod(pos_embed_shape) == d_embed or not have_pos_embed:
            self.pos_embed = 0
        else:
            self.pos_embed = nn.Parameter(param_init_fn(pos_embed_shape))
            if init_weight:
                nn.init.trunc_normal_(self.pos_embed, std=INIT_STD, a=-INIT_STD, b=INIT_STD)

        if stack_struct is not None:
            self.stack = MHAStack(
                d_embed=d_embed,
                d_kv_embed=d_embed,
                num_heads=num_heads,
                ff=ff,
                struct=stack_struct,
                dropout=dropout,
                d_adap=0,
                depth=stack_init_depth,
                do_weight_init=init_weight,
                have_crossnorm=have_crossnorm,
            )
        else:
            self.stack = lambda x: x

        match final_act:
            case "cat":
                self.final_act = lambda x: x.flatten(start_dim=1)
                self.linear = nn.Linear(input_numel.item(), n_class)
            case "pool":
                self.final_act = lambda x: x.mean(dim=1)
                self.linear = nn.Linear(d_embed, n_class)
            case "cls":
                self.final_act = lambda x: x[:, 0, :]
                self.linear = nn.Linear(d_embed, n_class)
            case _: raise NotImplementedError()
        
        if init_weight:
            nn.init.trunc_normal_(self.linear.weight, std=INIT_STD, a=-INIT_STD, b=INIT_STD)
            nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x):
        x = x + self.pos_embed

        x = rearrange(x, "B ... H -> B (...) H")
        x = self.stack(x)
        x = self.final_act(x)
        x = self.linear(x)
        
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
        multi_query_merge="seq", # cat, seq, ind

        d_embed=None,
        init_weight=False,
        embed_query=False,
        d_query_embed=None,
        have_ch_pos_embed=False,
        cat_ch_pos_embed=False,
        ch_pos_emb_sym=None, # None, "mirror",
        ch_order=["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6", "T6-O2", "A1-T3", "T3-C3", "C3-CZ", "C4-CZ", "T4-C4", "A2-T4", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"],
            
        clst_dim="", # TP
        clst_pos_embed_dim="N", # TPN
        n_clst=4,

        stack_struct="scf",
        num_heads=8,
        ff=4,
        dropout=0,
        have_crossnorm=True,

        across_pool_stack_struct="",
        n_ap_clst=0,
        ap_clst_dim="T", # T

        classifier_use_ap_clst=False,
        classifier_have_pos_embed=False,
        classifier_pos_embed_dim="TP", # TPN or TA
        classifier_stack_struct="sfsfsfsf",
        classifier_final_act="pool", # cat, pool, cls
        n_class=6
    ):
        if classifier_use_ap_clst:
            assert n_ap_clst > 0 and across_pool_stack_struct is not None
        super().__init__()

        self.extractor = LatentActivityExtractor(
            model=model,
            start=start,
            end=end,
            diffusion_t=diffusion_t,
            query=query
        )
        
        self.reducer = LatentActivityReducer(
            query=query,
            reduce=reduce,
            rescale=rescale,
            L=L,
            window_size=window_size,
            window_step=window_step,
            pool_merge=pool_merge,
            multi_query_merge=multi_query_merge
        )
        
        self.decoder = LatentActivityDecoder(
            d_model=model.d_model,
            d_kv_embed_factor=self.reducer.d_kv_embed_factor,
            n_layer=self.extractor.n_layer,
            d_embed=d_embed,
            init_weight=init_weight,
            
            n_query=self.reducer.n_query,
            embed_query=embed_query,
            d_query_embed=d_query_embed,
        
            have_ch_pos_embed=have_ch_pos_embed,
            cat_ch_pos_embed=cat_ch_pos_embed,
            ch_pos_emb_sym=ch_pos_emb_sym,
            ch_order=ch_order,
        
            clst_dim=clst_dim,
            clst_pos_embed_dim=clst_pos_embed_dim,
            n_clst=n_clst,
        
            n_pool=self.reducer.n_pool,
            pool_merge=pool_merge,
        
            multi_query_merge=multi_query_merge,
        
            stack_struct=stack_struct,
            num_heads=num_heads,
            ff=ff,
            dropout=dropout,
            have_crossnorm=have_crossnorm,

            across_pool_stack_struct=across_pool_stack_struct,
            n_ap_clst=n_ap_clst,
            ap_clst_dim=ap_clst_dim
        )

        self.use_rep_idx = 1 if classifier_use_ap_clst else 0
        self.classifier = TransformerClassifier(
            format="BTAH" if classifier_use_ap_clst else "BTPNH",
            n_tower=self.decoder.n_tower,
            n_pool=self.decoder.n_pool,
            n_clst=n_clst,
            n_ap_clst=self.decoder.n_ap_clst,
            d_embed=self.decoder.d_embed,
            have_pos_embed=classifier_have_pos_embed,
            pos_embed_dim=classifier_pos_embed_dim,
            stack_struct=classifier_stack_struct,
            num_heads=num_heads,
            ff=ff,
            dropout=dropout,
            have_crossnorm=have_crossnorm,
            stack_init_depth=self.decoder.cumulative_depth,
            final_act=classifier_final_act,
            init_weight=init_weight,
            n_class=n_class,
        )
    
    def forward(self, input, data_is_cached=False, rate=1):
        if not data_is_cached:
            latent_activity = self.extractor(input, rate=rate)
            tokens = self.reducer(latent_activity)
        else: 
            assert rate == 1
            tokens = input[:, self.extractor.start: self.extractor.end, :, :, :, :, :]
        rep = self.decoder(tokens)[self.use_rep_idx]
        cls = self.classifier(rep)
        return cls

    # def cache_la(self, input):
    #     return self.extractor.do_cache(input)
    
    def get_param_depth_by_name(self, name: str):
        if name.startswith("decoder.tower."):
            t_idx, n_idx, _, s_idx = name.split(".")[2:6]
            t_idx, n_idx, s_idx = list(map(int, (t_idx, n_idx, s_idx)))
            return self.decoder.tower[t_idx][n_idx].layer_depths[s_idx]
        # crossnorm?????
        elif name.startswith("decoder.across_pool_tower."):
            t_idx, n_idx, _, s_idx = name.split(".")[2:6]
            t_idx, n_idx, s_idx = list(map(int, (t_idx, n_idx, s_idx)))
            return self.decoder.across_pool_tower[t_idx][n_idx].layer_depths[s_idx]
        if not (name.startswith("decoder.tower.") or name.startswith("decoder.across_pool_tower.") or name.startswith("classifier.")):
            return 0
        

    def get_param_ls_by_depth(self):
        # if isinstance(self.classifier.stack, MHAStack):
        #     max_depth = self.classifier.stack.cumulative_depth
        # elif self.decoder.across_pool_tower[0][-1] is not None:
        #     max_depth = self.decoder.across_pool_tower[0][-1].cumulative_depth
        # else:
        #     max_depth = self.decoder.tower[0][-1].cumulative_depth

        # max_depth += 1

        depth_to_param_ls = {}

        flat_list = [tup for tup in zip(*self.decoder.tower)]
        if self.decoder.have_ap:
            flat_ap_list = [tup for tup in zip(*self.decoder.across_pool_tower)]
            flat_list = [val for tup in zip(flat_list, flat_ap_list) for val in tup]
        
        if isinstance(self.classifier.stack, MHAStack): flat_list.append([self.classifier.stack])

        cumulative_depth = 1
        for stack_tup in flat_list:
            for stack in stack_tup:
                for layer_ls in stack.layers_by_depth:
                    depth_to_param_ls[cumulative_depth] = []
                    for layer in layer_ls:
                        depth_to_param_ls[cumulative_depth].extend([param for param in layer.parameters()])
                    cumulative_depth += 1
        
        depth_to_param_ls[cumulative_depth] = list(self.classifier.linear.parameters())



        return depth_to_param_ls
