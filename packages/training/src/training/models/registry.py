from training.models.smp_strategy import SMPStrategy, SMP_MODELS

_REGISTRY = {
    **{name: SMPStrategy() for name in SMP_MODELS.keys()},
    # Add more strategies here
}

def get_model(config):
    model_name = config.model.model_name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not supported. Available {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name].build(config)
    