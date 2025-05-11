import os
from os.path import sep

DATA_BASE_PATH = "data"
DATASET_PATH = os.path.join(DATA_BASE_PATH, "dataset")
MODEL = "deepseek"

# Data Synthesize
SYNTHESIZED_DECOMPOSED_DATA_PATH = f"data/{MODEL}/synthesized_decomposed"
SYNTHESIZED_TOKEN_LABELING_DATA_PATH = f"data/{MODEL}/synthesized_token_labeling"
SYNTHESIZED_TOKEN_EXTRACTED_DATA_PATH = f"data/{MODEL}/token_extracted"
SYNTHESIZED_NEXT_QUERY_DATA_PATH = f"data/{MODEL}/synthesized_next_query"
SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH = f"data/{MODEL}/next_query_extracted"
SYNTHESIZED_NEGATIVE_SAMPLING_DATA_PATH = f"data/{MODEL}/negative_sampling"
SYNTHESIZED_NEGATIVE_SAMPLING_LABELED_DATA_PATH = f"data/{MODEL}/negative_sampling_labeled"
SYNTHESIZED_NEGATIVE_SAMPLING_EXTRACTED_DATA_PATH = (
    f"data/{MODEL}/negative_sampling_extracted"
)
EFFICIENT_RAG_LABELER_TRAINING_DATA_PATH = f"data/{MODEL}/efficient_rag/labeler"
EFFICIENT_RAG_FILTER_TRAINING_DATA_PATH = f"data/{MODEL}/efficient_rag/filter"
CORPUS_DATA_PATH = f"data/corpus"

SYNTHESIZED_SPAN_LABELING_DATA_PATH = f"data/{MODEL}/synthesized_span_labeling"

# Results
RETRIEVE_RESULT_PATH = f"results/{MODEL}/retrieve"

CONTINUE_TAG = "<CONTINUE>"
FINISH_TAG = "<FINISH>"
TERMINATE_TAG = "<TERMINATE>"

TAG_MAPPING = {
    CONTINUE_TAG: 0,
    TERMINATE_TAG: 1,
    FINISH_TAG: 2,
}
TAG_MAPPING_REV = {v: k for k, v in TAG_MAPPING.items()}

TAG_MAPPING_TWO = {
    CONTINUE_TAG: 0,
    TERMINATE_TAG: 1,
    FINISH_TAG: 0,
}
TAG_MAPPING_TWO_REV = {
    0: CONTINUE_TAG,
    1: TERMINATE_TAG,
}
TERMINATE_ID = TAG_MAPPING[TERMINATE_TAG]

# Special Tokens
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
PAD_TOKEN = "[PAD]"

MODEL_PATH = f"model_cache{sep}deberta-v3-large"
# MODEL_PATH = '/data1/Public/LLMs/deberta-v3-large'

MODEL_DICT = {
    "gpt-4.5": {
        "model_name": "gpt-4.5-preview",
        "api_key": "your_openai_api_key_here",  # 替换为实际 API Key
        "base_url": "https://api.openai.com/v1"  # OpenAI 默认端点
    },
    "gpt-4": {
        "model_name": "gpt-4-0125-preview",
        "api_key": "your_openai_api_key_here",
        "base_url": "https://api.openai.com/v1"
    },
    "gpt-4o": {
        "model_name": "gpt-4o",
        "api_key": "your_openai_api_key_here",
        "base_url": "https://api.openai.com/v1"
    },
    "gpt-4-turbo": {
        "model_name": "gpt-4-turbo",
        "api_key": "your_openai_api_key_here",
        "base_url": "https://api.openai.com/v1"
    },
    "llama": {
        "model_name": "Meta-Llama-3-70B-Instruct",
        "api_key": "your_llama_api_key_here",  # 如适用
        "base_url": "https://api.meta.com/llama"  # 假设的 Meta Llama API 端点
    },
    "llama-8B": {
        "model_name": "Meta-Llama-3-8B-Instruct",
        "api_key": None,  # 本地模型可能不需要 API Key
        "base_url": "/data1/Public/LLMs/Meta-Llama-3-8B-Instruct"  # 本地路径
    },
    "qwen2.5": {
        "model_name": "Qwen2.5-7B-Instruct",
        "api_key": "sk-",
        "base_url": "http://localhost:8000/v1"  # 本地路径
    },
    "deepseek": {
        "model_name": "deepseek-v3",
        "api_key": "sk-8aef464eb019408f94dc3cc5ef63d46e",
        "base_url":  "https://dashscope.aliyuncs.com/compatible-mode/v1"
    }
}
# DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# DEEPSEEK_API_KEY = "sk-56726d885ca64e959be6dfefc5b39312"
# DASHSCOPE_API_KEY = "sk-8aef464eb019408f94dc3cc5ef63d46e"
# DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

EMBEDDING_ALIAS = {
    "contriever": "contriever",
    "e5-base-v2": "e5-base",
    "e5-large-v2": "e5-large",
    "ada-002": "ada-002",
}
