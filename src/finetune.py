import torch
import lightning.pytorch as pl
import torch.utils
import torch.utils.data
from model.classifier_pl import PLClassifier
from dataloader.TUEVDataset import TUEVDataset
import os
from omegaconf import DictConfig
from hydra.utils import instantiate
import pickle
import random
import string

def entry(config: DictConfig):
    if config["caching"]["do_cache"]:
        cache_random_dir = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        cache_rand_root = os.path.join(config["caching"]["root"], cache_random_dir)
        while os.path.isdir(cache_rand_root):
            cache_random_dir = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            cache_rand_root = os.path.join(config["caching"]["root"], cache_random_dir)
    
    pl.seed_everything(**config["rng_seeding"])

    trainer = instantiate(config["trainer"])

    model = PLClassifier(
        diffusion_model_checkpoint=config["model"]["diffusion_model_checkpoint"],
        model_kwargs=config["model"]["model_kwargs"],
        ema_kwargs=config["model"]["ema_kwargs"],
        opt_kwargs=config["model"]["opt_kwargs"],
        sch_kwargs=config["model"]["sch_kwargs"],
        criterion_kwargs=config["model"]["criterion_kwargs"],
        fwd_with_noise=config["model"]["fwd_with_noise"],
        data_is_cached=config["caching"]["do_cache"],
        run_test_together=config["model"]["run_test_together"],
        cls_version=config["model"]["cls_version"],
        lrd_kwargs=config["model"]["lrd_kwargs"]
    )

    data_config = instantiate(config["data"])


    if config["caching"]["do_cache"]:
        raise NotImplementedError()
        # train_loader = torch.utils.data.DataLoader(
        #     TUEVDataset(
        #         os.path.join(data_config["root"], data_config["train_dir"]),
        #         schema=data_config["schema"]
        #     ), 
        #     batch_size=data_config["batch_size"],
        #     num_workers=1,
        #     shuffle=False
        # )
        
        # cache_train_dir = os.path.join(cache_rand_root, "train")
        # device = f"cuda:{config['trainer']['devices'][0]}"
        # model.model.to(device=device)
        # if not os.path.isdir(cache_train_dir):
        #     os.makedirs(cache_train_dir)
        # idx = 0
        # for batch_input in train_loader:
        #     batch = batch_input[0].to(device=device)
        #     label = batch_input[1].to(device=device)
        #     local = batch_input[2].to(device=device) if len(batch_input) > 2 else None

        #     computed = model.model.compute_tokens((batch, local))
        #     for c, l in zip(computed, label):
        #         with open(os.path.join(cache_train_dir, f"{idx}.pkl"), "wb") as f:
        #             pickle.dump({
        #                 "__cache_data__": c.cpu().numpy(),
        #                 "__cache_label__": l.cpu().numpy()
        #             }, f)
        #         idx += 1
        
        # val_loader = torch.utils.data.DataLoader(
        #     TUEVDataset(
        #         os.path.join(data_config["root"], data_config["val_dir"]),
        #         schema=data_config["schema"]
        #     ), 
        #     batch_size=data_config["batch_size"],
        #     num_workers=1,
        #     shuffle=False
        # )
        # cache_val_dir = os.path.join(cache_rand_root, "val")
        # if not os.path.isdir(cache_val_dir):
        #     os.makedirs(cache_val_dir)
        # idx = 0
        # for batch_input in val_loader:
        #     batch = batch_input[0].to(device=device)
        #     label = batch_input[1].to(device=device)
        #     local = batch_input[2].to(device=device) if len(batch_input) > 2 else None

        #     computed = model.model.compute_tokens((batch, local))
        #     for c, l in zip(computed, label):
        #         with open(os.path.join(cache_val_dir, f"{idx}.pkl"), "wb") as f:
        #             pickle.dump({
        #                 "__cache_data__": c.cpu().numpy(),
        #                 "__cache_label__": l.cpu().numpy()
        #             }, f)
        #         idx += 1
        

        # train_loader = torch.utils.data.DataLoader(
        #     TUEVDataset(
        #         cache_train_dir,
        #         schema=[
        #             ("__cache_data__", torch.float),
        #             ("__cache_label__", torch.long)
        #         ]
        #     ), 
        #     batch_size=data_config["batch_size"],
        #     num_workers=data_config["num_workers"],
        # )
        
        # val_loader = torch.utils.data.DataLoader(
        #     TUEVDataset(
        #         cache_val_dir,
        #         schema=[
        #             ("__cache_data__", torch.float),
        #             ("__cache_label__", torch.long)
        #         ]
        #     ), 
        #     batch_size=data_config["batch_size"],
        #     num_workers=data_config["num_workers"],
        # )

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
            schema=data_config["schema"],
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
        best_model = PLClassifier.load_from_checkpoint(
            trainer.checkpoint_callbacks[0].best_model_path
        )
        # pl.Trainer(devices=config["trainer"]["devices"][:1]).test(best_model, test_loader)
        trainer.test(best_model, test_loader)