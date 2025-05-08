from .aoai import AOAI
from .base import LanguageModel
from .deepseek import DeepSeek
from .llama import LlamaServer
from src.conf import MODEL_DICT


def get_model(model: str, **kwargs) -> LanguageModel:
    if "gpt" in model.lower():
        return AOAI(model=MODEL_DICT[model]['model_name'],
                    api_key=MODEL_DICT[model]['api_key'],
                    base_url=MODEL_DICT[model]['base_url'],
                    **kwargs)
    elif "deepseek" in model.lower():
        return DeepSeek(model=MODEL_DICT[model]['model_name'],
                        api_key=MODEL_DICT[model]['api_key'],
                        base_url=MODEL_DICT[model]['base_url'],
                        **kwargs)
    elif "llama" in model.lower():
        return LlamaServer(model=MODEL_DICT[model]['model_name'], **kwargs)
    else:
        raise NotImplementedError(f"Model {model} not implemented")
