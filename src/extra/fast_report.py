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
import matplotlib.pyplot as plt

# Use pytorch lightning for reporting, might be inaccurate due to ddp strategy
def entry(config: DictConfig):
    trainer = instantiate(config["trainer"])
    pl_cls = [None, None, PLClassifier_v2][config.get("pl_cls_version", 1)]
    model = pl_cls.load_from_checkpoint(config["checkpoint"])
    # print(model.hparams)
    # assert False
    # model._test_data_is_cached_for_fast_report = True

    data_config = instantiate(config["data"])

    test_loader = torch.utils.data.DataLoader(
        TUEVDataset(
            os.path.join(data_config["root"], data_config["test_dir"]),
            schema=data_config.get("test_schema", data_config["schema"]),
            stft_kwargs=data_config["stft_kwargs"]
        ), 
        # batch_size=data_config["batch_size"],
        # num_workers=data_config["num_workers"],
        batch_size=12,
        num_workers=2,
    )
    
    trainer.test(model, test_loader)