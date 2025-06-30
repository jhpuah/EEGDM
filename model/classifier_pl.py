import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from .classifier import Classifier as Classifier_v1
from .classifier_v2 import Classifier as Classifier_v2
from .classifier_v3 import Classifier as Classifier_v3
from .diffusion_model import Wavenet
from .diffusion_model_pl import PLDiffusionModel
from diffusers import DDPMScheduler
from ema_pytorch import EMA
from .util import setup_optimizer
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

        Classifier = [None, Classifier_v1, Classifier_v2, Classifier_v3][cls_version]

        diffusion_model: PLDiffusionModel = PLDiffusionModel.load_from_checkpoint(diffusion_model_checkpoint, map_location=self.device)
        self.model = Classifier(model=diffusion_model.ema.ema_model, **model_kwargs)
        self.model.model.eval()
        self.ema = EMA(
            self.model,
            ignore_startswith_names={"model"}, # ignore the diffusion backbone model in EMA
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

    def get_loss_pred_label(self, batch_input, use_ema=False, data_is_cached=False):
        model = self.ema if use_ema else self.model
        batch = batch_input[0]
        label = batch_input[1].view(-1)
        local_cond = batch_input[2] if len(batch_input) > 2 else None

        if not data_is_cached:
            noisy_signal = self.forward_sample(batch, force_zero_noise=use_ema)
            pred = model((noisy_signal, local_cond))
        else:
            pred = model(batch)

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