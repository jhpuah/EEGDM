import hydra
from omegaconf import DictConfig, OmegaConf


from src import preprocessing, pretrain, finetune, report, caching

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    # print("Received config:\n", OmegaConf.to_yaml(config))
    preprocessing_config = config.get("preprocessing", None)
    pretrain_config = config.get("pretrain", None)
    cache_config = config.get("cache", None)
    finetune_config = config.get("finetune", None)
    report_config = config.get("report", None)
    aux_config = config.get("aux", None)

    if preprocessing_config is not None:
        print("Enter preprocessing")
        preprocessing.entry(preprocessing_config)
    
    if pretrain_config is not None:
        print("Enter pretraining")
        pretrain.entry(pretrain_config)
    
    if cache_config is not None:
        print("Enter caching")
        caching.entry(cache_config)

    if finetune_config is not None:
        print("Enter finetuning")
        finetune.entry(finetune_config)

    if report_config is not None:
        print("Enter reporting")
        report.entry(report_config)

    if aux_config is not None:
        hydra.utils.instantiate(aux_config["target"])(aux_config) # horrible
# --config-name=file


if __name__ == "__main__":
    main()