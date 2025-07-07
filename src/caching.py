import torch
from dataloader.TUEVDataset import TUEVDataset
from hydra.utils import instantiate
from model.cclassifier import LatentActivityExtractor, LatentActivityReducer
from model.diffusion_model_pl import PLDiffusionModel
import os
from tqdm import tqdm
import pickle

@torch.no_grad()
def entry(config):
    # raise NotImplementedError()
    cache_config = config["cache"]    
    data_config = instantiate(config["data"])
    reduce_config = config["reduce"]
    
    cache_train_dir = os.path.join(cache_config["root"], "train")
    device = config['device']

    diffusion_model_checkpoint = config["diffusion_model_checkpoint"]

    diffusion_model = PLDiffusionModel.load_from_checkpoint(diffusion_model_checkpoint, map_location=device)
    model = torch.nn.Sequential(
        LatentActivityExtractor(
            model=diffusion_model.ema.ema_model,
            diffusion_t=config["diffusion_t"],
            query=reduce_config["query"]
        ),
        LatentActivityReducer(**reduce_config)
    )
    model.to(device=device)
    
    train_loader = torch.utils.data.DataLoader(
        TUEVDataset(
            os.path.join(data_config["root"], data_config["train_dir"]),
            schema=data_config["schema"],
            return_index=True,
        ), 
        batch_size=data_config["batch_size"],
        num_workers=4,
        shuffle=False
    )

    if not os.path.isdir(cache_train_dir):
        os.makedirs(cache_train_dir)
    for batch_input in tqdm(train_loader, total=len(train_loader.dataset) // data_config["batch_size"] + 1):
        batch = batch_input[0].to(device=device)
        label = batch_input[1].to(device=device)
        local = batch_input[2].to(device=device) if len(batch_input) > 3 else None
        index = batch_input[-1].to(device=device)

        computed = model((batch, local))
        for c, l, i in zip(computed, label, index):
            filename = os.path.basename(train_loader.dataset.files[i.item()])
            with open(os.path.join(cache_train_dir, filename), "wb") as f:
                pickle.dump({
                    "__cache_data__": c.cpu().numpy(),
                    "__cache_label__": l.cpu().numpy()
                }, f)

    val_loader = torch.utils.data.DataLoader(
        TUEVDataset(
            os.path.join(data_config["root"], data_config["val_dir"]),
            schema=data_config["schema"],
            return_index=True,
        ), 
        batch_size=data_config["batch_size"],
        num_workers=4,
        shuffle=False
    )
    cache_val_dir = os.path.join(cache_config["root"], "val")
    if not os.path.isdir(cache_val_dir):
        os.makedirs(cache_val_dir)
    for batch_input in val_loader:
        batch = batch_input[0].to(device=device)
        label = batch_input[1].to(device=device)
        local = batch_input[2].to(device=device) if len(batch_input) > 3 else None
        index = batch_input[-1].to(device=device)

        computed = model((batch, local))
        for c, l, i in zip(computed, label, index):
            filename = os.path.basename(train_loader.dataset.files[i.item()])
            with open(os.path.join(cache_val_dir, filename), "wb") as f:
                pickle.dump({
                    "__cache_data__": c.cpu().numpy(),
                    "__cache_label__": l.cpu().numpy()
                }, f)
    metadata = {
        "diffusion_model_checkpoint": diffusion_model_checkpoint,
        "diffusion_t": config["diffusion_t"],
        **reduce_config
    }

    with open(os.path.join(cache_config["root"], "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    
    # TODO copy test to the cached file