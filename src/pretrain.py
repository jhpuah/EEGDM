import torch
import lightning.pytorch as pl
import torch.utils
import torch.utils.data
from model.diffusion_model_pl import PLDiffusionModel
from dataloader.TUEVDataset import TUEVDataset
import os
from omegaconf import DictConfig
from hydra.utils import instantiate

def entry(config: DictConfig):
    pl.seed_everything(**config["rng_seeding"])

    trainer = instantiate(config["trainer"])

    model = PLDiffusionModel(
        model_kwargs=config["model"]["model_kwargs"],
        ema_kwargs=config["model"]["ema_kwargs"],
        noise_sch_kwargs=config["model"]["noise_sch_kwargs"],
        opt_kwargs=config["model"]["opt_kwargs"],
        gen_kwargs=instantiate(config["model"]["gen_kwargs"]),
        target_dist=None,
        kde_kwargs=None
    )

    data_config = instantiate(config["data"])
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
    
    # test_loader = torch.utils.data.DataLoader(
    #     TUEVDataset(
    #         os.path.join(data_config["root"], data_config["test_dir"]),
    #         schema=data_config["schema"]
    #     ), 
    #     batch_size=data_config["batch_size"],
    #     num_workers=data_config["num_workers"],
    # )
    
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(model, test_loader)
