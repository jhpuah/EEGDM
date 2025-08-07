import hydra
from omegaconf import DictConfig, OmegaConf
from src import preprocessing, pretrain, finetune, report, caching

# Usage:
# python main.py [preprocessing=?] [pretrain=?] [cache=?] [finetune=?] [report=?] [aux=?]
# replace "?" with config file name (without extenaion)
# the file must be put inside "conf", under the directory with the same name
#
# e.g.
#   python main.py pretrain=base
#       run pretraining with config specified in conf/pretrain/base.yaml
#
#   python main.py finetune=base finetune.rng_seeding.seed=10
#       run pretraining with config specified in conf/finetune/base.yaml, and set the rng seed to 10
# 
# see also: hydra documentation (https://hydra.cc/docs/intro/)
#
# "aux" config is special, main() will load a function specified in its "target" field
# and pass the config file to that function
# it is a quick and dirty way to add experiemnts that does not fit well to the established workflow
# 

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
        print("Entering aux:", aux_config["target"]["item"])
        hydra.utils.instantiate(aux_config["target"])(aux_config) # horrible
# --config-name=file


if __name__ == "__main__":
    main()