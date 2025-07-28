import torch
import numpy as np
from tqdm import tqdm
# from model.classifier_pl import PLClassifier
from .no_fusion import PLClassifier as PLClassifier_v2
from dataloader.TUEVDataset import TUEVDataset
from pyhealth.metrics.multiclass import multiclass_metrics_fn
from hydra.utils import instantiate
from sklearn import metrics
import gc

# TODO figure out how to distribute without repeated data
def entry(config):
    checkpoint = config["checkpoint"]
    if isinstance(checkpoint, str): checkpoint = [checkpoint]
    pl_cls=[None, None, PLClassifier_v2][config.get("pl_cls_version", 1)]
    
    dataset = TUEVDataset(
        config["data_dir"],
        schema=instantiate(config["schema"])
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False
    )

    data_count = len(dataset)

    all_result = []

    for c in tqdm(checkpoint):
        model = pl_cls.load_from_checkpoint(c, map_location=config["device"])
        y_true = np.zeros((data_count))
        y_prob = np.zeros((data_count, config["n_class"]))

        _idx = 0
        with torch.no_grad():
            for batch_input in tqdm(dataloader, total=data_count // config["batch_size"] + 1):
                batch_input = model.transfer_batch_to_device(batch_input, config["device"], 0)

                _, pred, _ = model.get_loss_pred_label(batch_input, use_ema=True, data_is_cached=False)

                _bs = pred.shape[0]
                y_true[_idx: _idx + _bs] = batch_input[1].flatten().cpu().numpy()
                y_prob[_idx: _idx + _bs, :] = torch.nn.functional.softmax(pred, dim=-1).cpu().numpy()
                _idx += _bs

        result = multiclass_metrics_fn(y_true, y_prob, metrics=config["metrics"])
        all_result.append(result)
        
        print(metrics.confusion_matrix(y_true, y_prob.argmax(axis=1)))
        
        del model, y_true, y_prob
        gc.collect()
        torch.cuda.empty_cache()

    for m in config["metrics"]:
        arr = np.array(list(map(lambda r: r[m], all_result))) * 100
        print(m, round(arr.mean().item(), 2), round(arr.std().item(), 2))