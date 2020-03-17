from logging import getLogger

logger = getLogger(__name__)
def load_best_model(model):
    return model.load(model.config.resource.model_best_conf_path, model.config.resource.model_best_weight_path)

def load_best_model_weight(model):
    return model.load(model.config.resource.model_best_conf_path, model.config.resource.model_best_weight_path)

def reload_best_model_weight_if_changed(model):
    if model.config.model.distributed:
        return load_best_model(model)
    else:
        logger.debug("Start reload the model if changed")
        digest = model.fetch_digest(model.config.resource.model_best_weight_path)
        if digest != model.digest:
            return load_best_model_weight(model)

        logger.debug("the best model is not changed")
        return False

def save_as_best_model(model):
    return model.save(model.config.resource.model_best_conf_path, model.config.resource.model_best_weight_path)
