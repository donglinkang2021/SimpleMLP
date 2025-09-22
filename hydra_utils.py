from omegaconf import OmegaConf

def suffix(_target_: str) -> str:
    return _target_.split(".")[-1]

def register_omegaconf_resolvers():
    OmegaConf.register_new_resolver("suffix", suffix)