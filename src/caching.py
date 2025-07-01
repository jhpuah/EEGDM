import torch
from dataloader import TUEVDataset

import os

def entry(config):
    # raise NotImplementedError()
    train_loader = torch.utils.data.DataLoader(
        TUEVDataset(
            os.path.join(data_config["root"], data_config["train_dir"]),
            schema=data_config["schema"],
            return_index=
        ), 
        batch_size=_cache_batch_size,
        num_workers=4,
        shuffle=False
    )

    cache_train_dir = os.path.join(cache_rand_root, "train")
    device = f"cuda:{config['trainer']['devices'][0]}"
    model.model.to(device=device)
    if not os.path.isdir(cache_train_dir):
        os.makedirs(cache_train_dir)
    idx = 0
    for batch_input in tqdm(train_loader, total=len(train_loader.dataset) // _cache_batch_size + 1):
        batch = batch_input[0].to(device=device)
        label = batch_input[1].to(device=device)
        local = batch_input[2].to(device=device) if len(batch_input) > 2 else None

        computed = model.model.compute_tokens((batch, local))
        for c, l in zip(computed, label):
            with open(os.path.join(cache_train_dir, f"{idx}.pkl"), "wb") as f:
                pickle.dump({
                    "__cache_data__": c.cpu().numpy(),
                    "__cache_label__": l.cpu().numpy()
                }, f)
            idx += 1

    val_loader = torch.utils.data.DataLoader(
        TUEVDataset(
            os.path.join(data_config["root"], data_config["val_dir"]),
            schema=data_config["schema"]
        ), 
        batch_size=_cache_batch_size,
        num_workers=4,
        shuffle=False
    )
    cache_val_dir = os.path.join(cache_rand_root, "val")
    if not os.path.isdir(cache_val_dir):
        os.makedirs(cache_val_dir)
    idx = 0
    for batch_input in val_loader:
        batch = batch_input[0].to(device=device)
        label = batch_input[1].to(device=device)
        local = batch_input[2].to(device=device) if len(batch_input) > 2 else None

        computed = model.model.compute_tokens((batch, local))
        for c, l in zip(computed, label):
            with open(os.path.join(cache_val_dir, f"{idx}.pkl"), "wb") as f:
                pickle.dump({
                    "__cache_data__": c.cpu().numpy(),
                    "__cache_label__": l.cpu().numpy()
                }, f)
            idx += 1