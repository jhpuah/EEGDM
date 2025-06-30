import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning_utilities.core.rank_zero import rank_zero_only
from .diffusion_model import Wavenet
from diffusers import DDPMScheduler
from ema_pytorch import EMA
from .util import setup_optimizer
import os
from einops import rearrange
import mne
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

class PLDiffusionModel(pl.LightningModule):
    def __init__(self, model_kwargs, ema_kwargs, noise_sch_kwargs, opt_kwargs, gen_kwargs, target_dist=None, kde_kwargs=None):
        if target_dist is not None: assert kde_kwargs is not None
        super().__init__()
        self.save_hyperparameters()

        self.model = Wavenet(**model_kwargs)
        self.ema = EMA(self.model, **ema_kwargs)
        self.noise_sch = DDPMScheduler(**noise_sch_kwargs)

        self.gen_save_dir = os.path.join(gen_kwargs["root"], gen_kwargs["save_dir"])
        self.pred_target = noise_sch_kwargs["prediction_type"]
        if not os.path.exists(self.gen_save_dir):
            os.makedirs(self.gen_save_dir)

    def configure_optimizers(self):
        return setup_optimizer(self.model, **self.hparams["opt_kwargs"])
        
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-4, epochs=epoch, steps_per_epoch=, pct_start=0.1)
        # return {"optimizer": optimizer, "scheduler": scheduler}

    def training_step(self, batch_input, batch_idx):
        # batch_input = self.transfer_batch_to_device(batch_input, self.device, batch_idx)

        batch = batch_input[0]
        label = batch_input[1].view(-1, 1)
        local_cond = batch_input[2] if len(batch_input) > 2 else None

        noisy_signal, noise, times = self.forward_sample(batch)
        target = self.get_pred_target(batch, noise, times)

        pred = self.model(noisy_signal, times, label, local_cond)

        loss = F.mse_loss(pred, target)
        self.log("train/mse_loss", loss, on_epoch=True, on_step=False)
        return loss
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update()
    
    @torch.no_grad()
    def validation_step(self, batch_input, batch_idx):
        # validate via MSE loss: does it make sense for diffusion model?
        
        batch = batch_input[0]
        label = batch_input[1].view(-1, 1)
        local_cond = batch_input[2] if len(batch_input) > 2 else None

        noisy_signal, noise, times = self.forward_sample(batch)
        target = self.get_pred_target(batch, noise, times)

        pred = self.ema(noisy_signal, times, label, local_cond)

        loss = F.mse_loss(pred, target)
        
        self.log("val/mse_loss", loss, on_epoch=True, on_step=False)

    # def test_step(self, batch_input, batch_idx):
    #     # validate via MSE loss: does it make sense for diffusion model?
        
    #     batch = batch_input[0]
    #     label = batch_input[1].view(-1, 1)

    #     noisy_signal, noise, times = self.forward_sample(batch)
    #     target = self.get_pred_target(batch, noise, times)

    #     pred = self.model(noisy_signal, times, label)

    #     loss = F.mse_loss(pred, target)
        
    #     self.log("test/mse_loss", loss, on_epoch=True, on_step=False)


    @torch.no_grad()
    @rank_zero_only
    def on_validation_epoch_end(self):
        # TODO can log to wandb too
        def save(x_t, idx, current_ep):
            x_t = x_t.squeeze(1).cpu().numpy()
            # FIXME all device have the same rng seed so they all generate the same signal...
            # fname = os.path.join(self.gen_save_dir, f"ep{current_ep}_{idx}_{str(self.device)}.fif")
            # imgname = os.path.join(self.gen_save_dir, f"ep{current_ep}_{idx}_{str(self.device)}.png")
            fname = os.path.join(self.gen_save_dir, f"ep{current_ep}_{idx}.fif")
            imgname = os.path.join(self.gen_save_dir, f"ep{current_ep}_{idx}.png")
            
            ch_names = self.hparams["gen_kwargs"]["ch_names"]
            if callable(self.hparams["gen_kwargs"]["inv_data_trans"]):
                x_t = self.hparams["gen_kwargs"]["inv_data_trans"](x_t)
            raw = mne.io.RawArray(
                    x_t * self.hparams["gen_kwargs"]["rescale"], 
                    mne.create_info(ch_names, self.hparams["gen_kwargs"]["sfreq"], "eeg", verbose="CRITICAL"),
                    verbose="CRITICAL"
                )
            raw.save(fname, overwrite=True, verbose="CRITICAL")
            raw.plot(duration=float("inf"), n_channels=len(ch_names), show=False).savefig(imgname)
            plt.close()
        
        n_sample = self.hparams["gen_kwargs"]["n_sample"]
        gen_cond = torch.arange(self.hparams["model_kwargs"]["n_class"], dtype=torch.long, device=self.device)\
            .repeat(n_sample)\
            .unsqueeze(-1)
        bs = len(gen_cond)

        x_t = torch.randn(bs, *self.hparams["gen_kwargs"]["shape"], device=self.device)
        
        timesteps = self.hparams["noise_sch_kwargs"]["num_train_timesteps"]

        for t in tqdm(range(timesteps - 1, 0, -1), desc="Sampling"):
            time = torch.ones(bs, 1, dtype=torch.long, device=self.device) * t
            pred = self.ema(x_t, time, gen_cond)

            x_t = self.noise_sch.step(pred, t,  x_t).prev_sample
            if self.hparams["gen_kwargs"]["save_intermediate"]:
                for i, _x_t in enumerate(x_t.chunk(n_sample, dim=0)):
                    save(_x_t, f"inter{i}_t{t}", self.current_epoch)
        for i, _x_t in enumerate(x_t.chunk(n_sample, dim=0)):
            save(_x_t, f"gen{i}", self.current_epoch)
        
        # Calculation of KL divergence
        if self.hparams["target_dist"] is not None: # means need to calculate kl div
            x_t_np = x_t.cpu().numpy()
            # use the model to approximate log p(x_0) is unreliable, due to the simplified objective
            gen_logprob = KernelDensity(**self.hparams["kde_kwargs"]).fit(x_t_np).score_samples(x_t_np)
            target_logprob = self.hparams["target_dist"].score_samples(x_t_np)
            kld = gen_logprob - target_logprob # math
            self.log("test/KLD", kld.mean().item())
        
    def forward_sample(self, batch, times=None):
        bs = batch.shape[0]
        noise = torch.randn_like(batch)
        if times is None:
            times = torch.randint(0, self.hparams["noise_sch_kwargs"]["num_train_timesteps"], (bs, 1), device=batch.device)

        noisy_signal = self.noise_sch.add_noise(batch, noise, times)
        return noisy_signal, noise, times
    
    def get_pred_target(self, batch, noise, times):
        match self.pred_target:
            case "sample":
                return batch
            case "epsilon":
                return noise
            case "v_prediction":
                return self.noise_sch.get_velocity(batch, noise, times)
        
    # def target_to_epsilon(self, batch, noised, pred, times):
    #     if self.pred_target == "epsilon":
    #         return pred
        
    #     alphas_cumprod = self.noise_sch.alphas_cumprod.to(device=self.device, dtype=batch.dtype)
    #     sqrt_alpha_prod = alphas_cumprod[times] ** 0.5
    #     sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    #     while len(sqrt_alpha_prod.shape) < len(batch.shape):
    #         sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    #     sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[times]) ** 0.5
    #     sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    #     while len(sqrt_one_minus_alpha_prod.shape) < len(batch.shape):
    #         sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
    #     match self.pred_target:
    #         case "sample":
    #             # noised = alpha * batch(pred) + one_minus * eps
    #             # eps = (noised - alpha * pred) / one_minus
    #             return (noised - sqrt_alpha_prod * pred) / sqrt_one_minus_alpha_prod
    #         case "v_prediction":
    #             # v(pred) = alpha * epsilon - one_minus * batch
    #             # eps = (pred + one_minus * batch) / alpha
    #             return (pred + sqrt_one_minus_alpha_prod * batch) / sqrt_alpha_prod
