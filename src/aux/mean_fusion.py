import torch
from torch import nn
from model.diffusion_model import Wavenet
from model.util import calc_diffusion_step_embedding
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
        if is_caching:
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

    def do_cache(self, input):
        return self(input, is_caching=True)

    def from_cache(self, cached_input):
        idx = torch.tensor(
            [["inter", "gate", "filter"].index(q) for q in self.query],
            dtype=torch.long,
            device=cached_input.device
        )
        return cached_input[:, :, :, idx, :, :]

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
        # match format:
        #     case "BTPNH":
        #         pos_embed_shape = [1, n_tower, n_pool, n_clst, d_embed]
        #     case "BTAH":
        #         pos_embed_shape = [1, n_tower, n_ap_clst, d_embed]
        #     case _: raise NotImplementedError()
        pos_embed_shape = [1, n_tower * n_pool * n_clst, d_embed]

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
        
        # self.decoder = LatentActivityDecoder(
        #     d_model=model.d_model,
        #     d_kv_embed_factor=self.reducer.d_kv_embed_factor,
        #     n_layer=self.extractor.n_layer,
        #     d_embed=d_embed,
        #     init_weight=init_weight,
            
        #     n_query=self.reducer.n_query,
        #     embed_query=embed_query,
        #     d_query_embed=d_query_embed,
        
        #     have_ch_pos_embed=have_ch_pos_embed,
        #     cat_ch_pos_embed=cat_ch_pos_embed,
        #     ch_pos_emb_sym=ch_pos_emb_sym,
        #     ch_order=ch_order,
        
        #     clst_dim=clst_dim,
        #     clst_pos_embed_dim=clst_pos_embed_dim,
        #     n_clst=n_clst,
        
        #     n_pool=self.reducer.n_pool,
        #     pool_merge=pool_merge,
        
        #     multi_query_merge=multi_query_merge,
        
        #     stack_struct=stack_struct,
        #     num_heads=num_heads,
        #     ff=ff,
        #     dropout=dropout,
        #     have_crossnorm=have_crossnorm,

        #     across_pool_stack_struct=across_pool_stack_struct,
        #     n_ap_clst=n_ap_clst,
        #     ap_clst_dim=ap_clst_dim
        # )
        self.decoder = lambda t: rearrange(t, "B n ... H -> B n (...) H").mean(dim=1)

        # self.use_rep_idx = 1 if classifier_use_ap_clst else 0
        self.classifier = TransformerClassifier(
            format="BTPNH",
            n_tower=1,
            n_pool=5,
            n_clst=22,
            n_ap_clst=0,
            d_embed=model.d_model,
            have_pos_embed=classifier_have_pos_embed,
            pos_embed_dim=classifier_pos_embed_dim,
            stack_struct=classifier_stack_struct,
            num_heads=num_heads,
            ff=ff,
            dropout=dropout,
            have_crossnorm=have_crossnorm,
            stack_init_depth=0,
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
            tokens = input
        rep = self.decoder(tokens)#[self.use_rep_idx]
        cls = self.classifier(rep)
        return cls

    def cache_la(self, input):
        return self.extractor.do_cache(input)

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
# from model.cclassifier import Classifier as Classifier_v1
Classifier_v1=Classifier
from model.diffusion_model_pl import PLDiffusionModel
from diffusers import DDPMScheduler
from ema_pytorch import EMA
from model.util import setup_optimizer
import os
from einops import rearrange
import mne
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy, MulticlassCohenKappa, MulticlassF1Score, MulticlassRecall, MulticlassConfusionMatrix
from torchmetrics import MetricCollection
import wandb
from copy import deepcopy

class CustomCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        gamma: float = 0,
        # is_binary: bool = False
    ):
        super().__init__()
        self.weight = weight
        match reduction:
            case "mean": self.reduce_fn = torch.mean
            case "sum": self.reduce_fn = torch.sum
            case _: raise NotImplementedError()
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        # self.is_binary = is_binary
        # if self.is_binary: assert label_smoothing == 0
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, weight=self.weight, reduction="none", label_smoothing=self.label_smoothing)
        if self.gamma == 0: return self.reduce_fn(ce_loss)

        prob = F.softmax(pred, dim=-1) * F.one_hot(target, num_classes=pred.shape[-1])
        confidence = prob.sum(dim=-1, keepdim=True)
        focal_weight = (1 - confidence) ** self.gamma
        ce_loss = focal_weight * ce_loss

        return self.reduce_fn(ce_loss)

class PLClassifier(pl.LightningModule):
    def __init__(self, diffusion_model_checkpoint, model_kwargs, ema_kwargs, opt_kwargs, sch_kwargs, criterion_kwargs, fwd_with_noise, data_is_cached, run_test_together=False, cls_version=1, lrd_kwargs=None):
        super().__init__()
        self.save_hyperparameters()

        # print(self.hparams)

        Classifier = [None, Classifier_v1][cls_version]

        diffusion_model: PLDiffusionModel = PLDiffusionModel.load_from_checkpoint(diffusion_model_checkpoint, map_location=self.device)
        self.model = Classifier(model=diffusion_model.ema.ema_model, **model_kwargs)
        self.ema = EMA(
            self.model,
            ignore_startswith_names={"extractor", "reducer"}, # ignore the diffusion backbone model in EMA
            **ema_kwargs
        )
        self.noise_sch = diffusion_model.noise_sch

        self.val_metrics = MetricCollection(
            {
                "bacc": MulticlassAccuracy(num_classes=6, average="macro", validate_args=False), # B C, B
                # "bacc1": MulticlassRecall(num_classes=6, average="macro", validate_args=False), # B C, B
                "kappa": MulticlassCohenKappa(num_classes=6, weights=None, validate_args=False),
                "wf1": MulticlassF1Score(num_classes=6, average="weighted", validate_args=False),
            },
            prefix="val/",
        )
        self.test_metrics = self.val_metrics.clone(prefix="test/")
        
        # deadlock
        # self.train_metrics = self.val_metrics.clone(prefix="train/")
        
        self.criterion = CustomCrossEntropyLoss(**criterion_kwargs)
        
        if fwd_with_noise:
            assert not data_is_cached
            self.noise_fn = torch.randn_like
        elif fwd_with_noise is None:
            self.noise_fn = None
        else:
            self.noise_fn = torch.zeros_like

    def configure_optimizers(self):
        if self.hparams["lrd_kwargs"] is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), **self.hparams["opt_kwargs"])
        else:
            if self.hparams["lrd_kwargs"].get("use_new_setup", False):
                no_wd = self.hparams["lrd_kwargs"].get("no_wd", [])
                bias_1dim_no_wd = self.hparams["lrd_kwargs"].get("bias_1dim_no_wd", False)
                
                def should_have_decay(name, param):
                    if name in no_wd: return False
                    if bias_1dim_no_wd:
                        if param.ndim <= 1 or name.endswith(".bias"):
                            return False
                    return True

                # assert "lr_decay" not in self.hparams["lrd_kwargs"]
                lr_decay_groups = self.hparams["lrd_kwargs"].get("lr_decay", [1])
                lr_decay_rate = lr_decay_groups[0]
                lr_decay_groups = lr_decay_groups[1:]

                def get_lrd_rate(name):
                    _lrd_rate = lr_decay_rate
                    for group in lr_decay_groups:
                        for prefix in group:
                            if name.startswith(prefix): return _lrd_rate
                        _lrd_rate *= lr_decay_rate
                    return 1

                spec_to_param_ls = {}
                default_spec = (True, 1)

                for name, param in self.model.named_parameters():
                    spec_wd = should_have_decay(name, param)
                    spec_lrd = get_lrd_rate(name)

                    spec = (spec_wd, spec_lrd)
                    if spec not in spec_to_param_ls.keys():
                        spec_to_param_ls[spec] = []
                    spec_to_param_ls[spec].append(param)
                
                optimizer = torch.optim.AdamW(spec_to_param_ls[default_spec], **self.hparams["opt_kwargs"])            
                spec_to_param_ls.pop(default_spec)
                for spec, param_ls in spec_to_param_ls.items():
                    optim_defaults = deepcopy(optimizer.defaults)
                    if not spec[0]:
                        optim_defaults["weight_decay"] = 0
                    optim_defaults["lr_decay"] = spec[1]
                    optim_defaults["lr"] *= spec[1]
                    optimizer.add_param_group({
                        "params": param_ls,
                        **optim_defaults
                    })

            else: # old, simple setup                        
                param_without_decay = [param for name, param in self.model.named_parameters() if name in self.hparams["lrd_kwargs"]["no_wd"]]
                param_with_decay = [param for name, param in self.model.named_parameters() if name not in self.hparams["lrd_kwargs"]["no_wd"]]
                optimizer = torch.optim.AdamW(param_with_decay, **self.hparams["opt_kwargs"])            

                optim_defaults = deepcopy(optimizer.defaults)
                optim_defaults["weight_decay"] = 0
                optimizer.add_param_group({
                    "params": param_without_decay,
                    **optim_defaults
                })
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            **self.hparams["sch_kwargs"]
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": None,
            "strict": False,
            "name": None,
        }

        return [optimizer], [lr_scheduler_config]

    def training_step(self, batch_input, batch_idx):
        loss, pred, label = self.get_loss_pred_label(batch_input, use_ema=False, data_is_cached=self.hparams["data_is_cached"])
        self.log("train/loss", loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True, add_dataloader_idx=False, batch_size=pred.shape[0])

        # self.train_metrics.update(pred, label)
        return loss
    
    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_closure = None,
    ):
        super().optimizer_step(epoch=epoch, batch_idx=batch_idx, optimizer=optimizer, optimizer_closure=optimizer_closure)
        self.ema.update()
        
        if self.hparams["lrd_kwargs"]is not None and self.hparams["lrd_kwargs"].get("use_new_setup", False):
            for param_group in optimizer.param_groups:
                if "lr_decay" in param_group.keys():
                    param_group["lr"] *= param_group["lr_decay"]
        

    @torch.no_grad()
    def validation_step(self, batch_input, batch_idx, dataloader_idx=0):
        loss, pred, label = self.get_loss_pred_label(batch_input, use_ema=True, data_is_cached=self.hparams["data_is_cached"])

        if self.hparams["run_test_together"] and dataloader_idx > 0:
            self.log("test/loss", loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True, add_dataloader_idx=False, batch_size=pred.shape[0])
            self.test_metrics.update(pred, label)
            # self.test_conf_mat.update(pred, label)
        else:
            self.log("val/loss", loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True, add_dataloader_idx=False, batch_size=pred.shape[0])
            self.val_metrics.update(pred, label)
            # self.val_conf_mat.update(pred, label)
        
        return loss
    
    @torch.no_grad()
    def test_step(self, batch_input, batch_idx):
        loss, pred, label = self.get_loss_pred_label(batch_input, use_ema=True, data_is_cached=False)
        
        self.log("test/loss", loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True, add_dataloader_idx=False, batch_size=pred.shape[0])
        self.test_metrics.update(pred, label)
        # self.test_conf_mat.update(pred, label)
        return loss
    

    def on_train_epoch_end(self):
        for i, lr in enumerate(self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()):
            self.log(f"train/lr_{i}", lr, on_epoch=True, on_step=False, sync_dist=True)
    
        # Uncomment this for ✨✨✨D E A D L O C K✨✨✨
        # self.log_dict(self.train_metrics.compute(), sync_dist=True, prog_bar=True)
        # self.train_metrics.reset()
    
    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), sync_dist=True, prog_bar=True)
        self.val_metrics.reset()
            
        # self.val_conf_mat.compute()
        # wandb.log({"val/conf_mat": self.val_conf_mat.plot()[0]})
        # self.val_conf_mat.reset()
    
        if self.hparams["run_test_together"]:
            self.log_dict(self.test_metrics.compute(), sync_dist=True, prog_bar=True)
            self.test_metrics.reset()

            # self.test_conf_mat.compute()
            # wandb.log({"test/conf_mat": self.test_conf_mat.plot()[0]})
            # self.test_conf_mat.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), sync_dist=True, prog_bar=True)
        self.test_metrics.reset()

    def get_loss_pred_label(self, batch_input, use_ema=False, data_is_cached=False, rate=1):
        assert rate == 1 or not data_is_cached
        model = self.ema if use_ema else self.model
        batch = batch_input[0]
        label = batch_input[1].view(-1)
        local_cond = batch_input[2] if len(batch_input) > 2 else None

        if not data_is_cached:
            noisy_signal = self.forward_sample(batch, force_zero_noise=use_ema)
            pred = model((noisy_signal, local_cond), data_is_cached=data_is_cached, rate=rate)
        else:
            pred = model(batch, data_is_cached=data_is_cached)

        return self.criterion(pred, label), pred, label

    def forward_sample(self, batch, force_zero_noise=None):
        if self.noise_fn is not None:
            if force_zero_noise:
                noise_fn = torch.zeros_like
            else:
                noise_fn = self.noise_fn
            bs = batch.shape[0]
            noise = noise_fn(batch)
            times = torch.ones((bs, 1), device=batch.device,  dtype=torch.long) * self.hparams["model_kwargs"]["diffusion_t"]
            noisy_signal = self.noise_sch.add_noise(batch, noise, times)
        else:
            noisy_signal = batch
        return noisy_signal



import torch
import lightning.pytorch as pl
import torch.utils
import torch.utils.data
# from model.classifier_pl import PLClassifier
# from model.cclassifier_pl import PLClassifier as PLClassifier_v2
PLClassifier_v2=PLClassifier
from dataloader.TUEVDataset import TUEVDataset
import os
from omegaconf import DictConfig
from hydra.utils import instantiate
import pickle
import random
import string
from tqdm import tqdm

def entry(config: DictConfig):
    pl.seed_everything(**config["rng_seeding"])

    trainer = instantiate(config["trainer"])
    data_is_cached = config.get("data_is_cached", False)
    if data_is_cached:
        metadata_provided = {
            "diffusion_model_checkpoint": config["model"]["diffusion_model_checkpoint"],
            "diffusion_t": config["model"]["model_kwargs"]["diffusion_t"],
            "query": config["model"]["model_kwargs"]["query"],
            "reduce": config["model"]["model_kwargs"]["reduce"],
            "rescale": config["model"]["model_kwargs"]["rescale"],
            "L": config["model"]["model_kwargs"]["L"],
            "window_size": config["model"]["model_kwargs"]["window_size"],
            "window_step": config["model"]["model_kwargs"]["window_step"],
            "pool_merge": config["model"]["model_kwargs"]["pool_merge"],
            "multi_query_merge": config["model"]["model_kwargs"]["multi_query_merge"],
        }
        
        with open(os.path.join(config["data"]["root"], "metadata.pkl"), "rb") as m:
            metadata = pickle.load(m)
        assert metadata.keys() == metadata_provided.keys()
        for k in metadata_provided.keys(): assert metadata[k] == metadata_provided[k]

    pl_cls = [None, PLClassifier, PLClassifier_v2][config.get("pl_cls_version", 1)]
    model = pl_cls(
        diffusion_model_checkpoint=config["model"]["diffusion_model_checkpoint"],
        model_kwargs=config["model"]["model_kwargs"],
        ema_kwargs=config["model"]["ema_kwargs"],
        opt_kwargs=config["model"]["opt_kwargs"],
        sch_kwargs=config["model"]["sch_kwargs"],
        criterion_kwargs=config["model"]["criterion_kwargs"],
        fwd_with_noise=config["model"]["fwd_with_noise"],
        data_is_cached=data_is_cached,
        run_test_together=config["model"]["run_test_together"],
        cls_version=config["model"]["cls_version"],
        lrd_kwargs=config["model"]["lrd_kwargs"]
    )

    data_config = instantiate(config["data"])


    if data_is_cached:
        train_loader = torch.utils.data.DataLoader(
            TUEVDataset(
                os.path.join(data_config["root"], data_config["train_dir"]),
                schema=data_config["schema"],
            ), 
            batch_size=data_config["batch_size"],
            num_workers=data_config["num_workers"],
        )

        val_loader = torch.utils.data.DataLoader(
            TUEVDataset(
                os.path.join(data_config["root"], data_config["val_dir"]),
                schema=data_config["schema"],
            ), 
            batch_size=data_config["batch_size"],
            num_workers=data_config["num_workers"],
        )

    else:
        train_loader = torch.utils.data.DataLoader(
            TUEVDataset(
                os.path.join(data_config["root"], data_config["train_dir"]),
                schema=data_config["schema"],
                stft_kwargs=data_config["stft_kwargs"]
            ), 
            batch_size=data_config["batch_size"],
            num_workers=data_config["num_workers"],
        )
        
        val_loader = torch.utils.data.DataLoader(
            TUEVDataset(
                os.path.join(data_config["root"], data_config["val_dir"]),
                schema=data_config["schema"],
                stft_kwargs=data_config["stft_kwargs"]
            ), 
            batch_size=data_config["batch_size"],
            num_workers=data_config["num_workers"],
        )

    test_loader = torch.utils.data.DataLoader(
        TUEVDataset(
            os.path.join(data_config["root"], data_config["test_dir"]),
            schema=data_config.get("test_schema", data_config["schema"]),
            stft_kwargs=data_config["stft_kwargs"]
        ), 
        # batch_size=data_config["batch_size"],
        # num_workers=data_config["num_workers"],
        batch_size=16,
        num_workers=2,
    )
    
    if config["model"]["run_test_together"]:
        trainer.fit(model, train_loader, [val_loader, test_loader])
    else:
        trainer.fit(model, train_loader, val_loader)
        best_model = pl_cls.load_from_checkpoint(
            trainer.checkpoint_callbacks[0].best_model_path
        )
        # pl.Trainer(devices=config["trainer"]["devices"][:1]).test(best_model, test_loader)
        trainer.test(best_model, test_loader)