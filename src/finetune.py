import torch
import lightning.pytorch as pl
import torch.utils
import torch.utils.data
from model.classifier_pl import PLClassifier as PLClassifier_v2
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

    pl_cls = [None, None, PLClassifier_v2][config.get("pl_cls_version", 1)]
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