import torch
from torch import nn
import numpy as np
from .s4standalone import FFTConv
from einops import rearrange
from .util import calc_diffusion_step_embedding
from typing import Literal

class WavenetBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_cond,
        local_cond_ch,
        local_cond_upsample=[8, 8],
        dropout=0.0,
        tie_dropout=False,
        is_last=False,
        self_gated=False,
        **layer_args,
    ):
        super().__init__()
        self.d_model = d_model
        self.is_last = is_last

        cond = nn.Linear(d_cond, d_model)
        cond.weight.data.zero_()
        cond.bias.data.zero_()
        self.cond_linear = cond

        self.prenorm1 = nn.LayerNorm(d_model)
        
        self.self_gated = self_gated
        if not self_gated:
            d_ssm = d_model * 2
            self.pre_layer = nn.Linear(d_model, d_model * 2)
        else:
            d_ssm = d_model
            self.pre_layer = nn.Identity()
        self.layer = FFTConv(d_model=d_ssm, transposed=False, dropout=dropout, tie_dropout=tie_dropout, activation=None, **layer_args)

        if local_cond_ch > 0:
            local_cond_conv = []
            for s in local_cond_upsample:
                conv_trans2d = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
                conv_trans2d = nn.utils.parametrizations.weight_norm(conv_trans2d)
                torch.nn.init.kaiming_normal_(conv_trans2d.weight)
                local_cond_conv.append(conv_trans2d)
                local_cond_conv.append(nn.LeakyReLU(negative_slope=0.4))
            self.local_cond_conv = nn.Sequential(*local_cond_conv)

            local_cond_linear = nn.Linear(local_cond_ch, d_ssm)
            nn.utils.parametrizations.weight_norm(local_cond_linear)
            nn.init.kaiming_normal_(local_cond_linear.weight)
            self.local_cond_linear = local_cond_linear


        if not self.is_last:
            self.linear_next = nn.Linear(d_model, d_model)
        self.linear_skip = nn.Linear(d_model, d_model)

    def forward(self, x, c, lc, query: Literal["gate", "filter", "inter"] | list = None, skip_skip=False, rate=1):
        assert x.shape[-1] == self.d_model
                
        skip = x

        cond_bias = self.cond_linear(c)

        x = self.prenorm1(x)
        x = x + cond_bias
        x = self.pre_layer(x)
        x, _ = self.layer(x, rate=rate)
        
        if lc is not None: # B 1 F W
            lc = self.local_cond_conv(lc)
            assert lc.shape[-1] >= x.shape[1]
            lc = rearrange(lc[:, :, :, :x.shape[1]], "B 1 F M -> B M F")
            lc = self.local_cond_linear(lc)
            x = x + lc
        
        if self.self_gated:
            gate = filter = x
        else:
            gate, filter = x.chunk(2, dim=-1)
        gate = torch.nn.functional.sigmoid(gate)
        filter = torch.nn.functional.tanh(filter)

        inter = gate * filter

        out_skip = None if skip_skip else self.linear_skip(inter)

        if not self.is_last:
            out_next = self.linear_next(inter) + skip
        else:
            out_next = None

        if query is not None:
            if isinstance(query, str): query = [query]
            query_result = []
            for q in query:
                match q:
                    case "gate": r = gate
                    case "filter": r = filter
                    case "inter": r = inter
                    case _: raise NotImplementedError()
                query_result.append(r)
            return out_next, out_skip, query_result
        return out_next, out_skip

class Wavenet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        d_model=128,
        d_state=128,
        n_layer=20,
        n_ssm=None,
        kernel_init="diag",
        kernel_mode="diag",
        bidirectional=True,
        d_cond=512,
        d_cond_embed=128,
        local_cond_ch=0,
        local_cond_upsample=[8, 8],
        n_class=22,
        have_null_class=False, # -1 = null class
        self_gated=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_embed = d_cond_embed
        self.have_null_class = have_null_class

        self.in_layer = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.ReLU()
        )

        self.layers = nn.ModuleList([
            WavenetBlock(
                d_model=d_model,
                d_state=d_state,
                d_cond=d_cond,
                local_cond_ch=local_cond_ch,
                local_cond_upsample=local_cond_upsample,
                n_ssm=n_ssm,
                init=kernel_init,
                mode=kernel_mode,
                bidirectional=bidirectional,
                is_last= _ == n_layer - 1,
                self_gated=self_gated
            ) for _ in range(n_layer)
        ])
        
        self.out_layer = nn.ModuleList([
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, in_channels)
        ])
        # kaiming_normal initialization
        nn.init.kaiming_normal_(self.out_layer[0].weight)

        # zero init
        self.out_layer[2].weight.data.zero_()
        self.out_layer[2].bias.data.zero_()

        self.t_embed_in = nn.Sequential(
            nn.Linear(d_cond_embed, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
            nn.SiLU()
        )
        
        if have_null_class:
            n_class += 1
        self.n_class = n_class

        self.label_embed = nn.Embedding(n_class, d_cond_embed)
        self.label_embed_in = nn.Sequential(
            nn.Linear(d_cond_embed, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
            nn.SiLU()
        )


    def forward(self, x, diffusion_steps, cond, local_cond=None):
        cond = self.calc_cond(diffusion_steps, cond)

        x = rearrange(x, "B C L -> B L C")
        x = self.in_layer(x)
        
        all_skip = 0
        for l in self.layers:
            x, skip = l(x, cond, local_cond)
            all_skip += skip

        for l in self.out_layer:
            all_skip = l(all_skip)

        all_skip = rearrange(all_skip, "B L C -> B C L")
        return all_skip
    
    def calc_cond(self, diffusion_steps, cond):
        t_embed = calc_diffusion_step_embedding(diffusion_steps, self.d_embed)
        
        t_embed = self.t_embed_in(t_embed).unsqueeze(1) # B 1 D

        if self.have_null_class:
            cond += 1 # -1 (null class) -> 0, 0 -> 1 ...
        cond = self.label_embed(cond)
        cond = self.label_embed_in(cond)
        
        return cond + t_embed

