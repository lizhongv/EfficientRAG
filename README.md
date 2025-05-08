## Installation 
```bash
# 1 git 
# export https_proxy=http://agent.baidu.com:8891
git clone https://github.com/lizhongv/EfficientRAG.git
cd EfficientRAG

# 2 uv env
pip install uv
uv venv effrag --python 3.10 && source effrag/bin/activate && uv pip install --upgrade pip

# 3 install vllm
export VLLM_VERSION=0.6.1.post1
export PYTHON_VERSION=310

pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118

pip install setuptools

# 4 install others
pip install -r requirements.txt

# 5 download spacy
python -m spacy download en_core_web_sm
```
## Preparation

download model
```bash
cd EfficientRAG
mkdir model_cache

# pip install huggingface_hub
# huggingface-cli download --resume-download facebook/contriever-msmarco --local-dir contriever-msmarco --local-dir-use-symlinks False
# huggingface-cli download --resume-download microsoft/deberta-v3-large --local-dir deberta-v3-large --local-dir-use-symlinks False

# pip install modelscope
# modelscope download --model zl2272001/deberta-v3-large  --local_dir ./deberta-v3-large
# modelscope download --model zl2272001/contriever-msmarco  --local_dir ./contriever-msmarco
```


Prepare the corpus by extract documents
```bash
# 2. download all the processed data 
modelscope download --dataset zl2272001/EfficientRAG  --local_dir ./
tar -zxvf data.tar.gz # 解压

# （1）Unify the data format
# python src/retrievers/data_processing.py

# （2） Prepare the corpus by extract documents 
# python src/retrievers/multihop_data_extrator.py --dataset hotpotQA
# python src/retrievers/multihop_data_extrator.py --dataset 2WikiMQA
# python src/retrievers/multihop_data_extrator.py --dataset musique
```

construct embedding
```bash
python src/retrievers/passage_embedder.py \
    --passages data/corpus/hotpotQA/corpus.jsonl \
    --output_dir data/corpus/hotpotQA/contriever \
    --model_type contriever 

python src/retrievers/passage_embedder.py \
    --passages data/corpus/2WikiMQA/corpus.jsonl \
    --output_dir data/corpus/2WikiMQA/contriever \
    --model_type contriever 

python src/retrievers/passage_embedder.py \
    --passages data/corpus/musique/corpus.jsonl \
    --output_dir data/corpus/musique/contriever \
    --model_type contriever 
```

## Data Construction
1. Query Decompose
   
```bash
python src/data_synthesize/query_decompose.py \
    --dataset 2WikiMQA \
    --split train \
    --model deepseek \
    --workers 10 \
    --ending  500
```

2. Token Labeling

```bash
python src/data_synthesize/token_labeling.py \
    --dataset 2WikiMQA \
    --split train \
    --model deepseek \
    --workers 10 
```

3. Token Extraction
```bash
python src/data_synthesize/token_extraction.py \
    --data_path data/deepseek/synthesized_token_labeling/2WikiMQA/train.jsonl \
    --save_path data/deepseek/token_extracted/2WikiMQA/train.jsonl \
    --verbose
```
4. Next Query construction
```bash
python src/data_synthesize/next_hop_query_construction.py \
    --dataset 2WikiMQA \
    --split train \
    --model deepseek
```

4. Next Query Filtering
```bash
python src/data_synthesize/next_hop_query_filtering.py \
    --data_path data/synthesized_next_query/2WikiMQA/train.jsonl \
    --save_path data/next_query_extracted/2WikiMQA/train.jsonl \
    --verbose
```

Negative Sampling
```bash
python src/data_synthesize/negative_sampling.py \
    --dataset 2WikiMQA \
    --split train \
    --retriever contriever

python src/data_synthesize/negative_sampling_labeled.py \
    --dataset 2WikiMQA \
    --split train \
    --model deepseek

python src/data_synthesize/negative_token_extraction.py \
    --dataset 2WikiMQA \
    --split train \
    --verbose
```



retrieve
```bash
python src/efficientrag_retrieve.py \
    --dataset hotpotQA \
    --retriever contriever \
    --labels 2 \
    --labeler_ckpt "/data0/lizhong/multi_hop_rag/EfficientRAG/saved_models/labeler_two/result/checkpoint-5584" \
    --filter_ckpt "/data0/lizhong/multi_hop_rag/EfficientRAG/saved_models/filter/result/checkpoint-9076" \
    --topk 10 \
```


## Data Construction
```bash
# 在 src/language_models/aoai.py 中添加
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=your_base_url_here

# 在 src/language_models/deepseek.py 中添加
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_API_KEY = ""
DASHSCOPE_API_KEY = ""
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# 1. Query Decompose  
python src/data_synthesize/query_decompose.py \
    --dataset hotpotQA \
    --split train \
    --model gpt-4o \
    --ending 10
# musique, hotpotQA, 2WikiMQA

# 2. Token Labeling
python src/data_synthesize/token_labeling.py \
    --dataset hotpotQA \
    --split train \
    --model llama3
python src/data_synthesize/token_extraction.py \
    --data_path data/synthesized_token_labeling/hotpotQA/train.jsonl \
    --save_path data/token_extracted/hotpotQA/train.jsonl \
    --verbose

# 3. Next Query Filtering
python src/data_synthesize/next_hop_query_construction.py \
    --dataset hotpotQA \
    --split train \
    --model llama
python src/data_synthesize/next_hop_query_filtering.py \
    --data_path data/synthesized_next_query/hotpotQA/train.jsonl \
    --save_path data/next_query_extracted/hotpotQA/train.jsonl \
    --verbose

# 4. Negative Sampling
python src/data_synthesize/negative_sampling.py \
    --dataset hotpotQA \
    --split train \
    --retriever contriever
python src/data_synthesize/negative_sampling_labeled.py \
    --dataset hotpotQA \
    --split train \
    --model llama
python src/data_synthesize/negative_token_extraction.py \
    --dataset hotpotQA \
    --split train \
    --verbose

# 5 Training Data
python src/data_synthesize/training_data_synthesize.py \
    --dataset hotpotQA \
    --split train
```

## Training
```bash
# 4. Training Filter model
python src/efficient_rag/filter_training.py \
    --dataset hotpotQA \
    --save_path saved_models/filter \
    --model_name_or_path ../deberta-v3-large \   

# 5. Training Labeler mode
nohup python -u src/efficient_rag/labeler_training.py \
    --dataset hotpotQA \
    --tags 2 \
    --model_name_or_path ../deberta-v3-large &

# 5.1 check log 
tensorboard --logdir=saved_models/filter/filter_20250401_043856/log --host 0.0.0.0 --port 31827

# 6. Construct embedding

```

## wandb 使用
```bash
# 安装
pip install wandb
# 官网注册
https://wandb.ai/site
# 登录 API 密钥
https://wandb.ai/site
# 初始化项目
os.environ["WANDB_PROJECT"] = "EfficientRAG_filter"
TrainingArguments 设置 run_name
```

## 数据合成
```bash
# Token Labeling Prompt  

You have been assigned an information extraction task.  
Your mission is to extract the words from a given paragraph so that others can answer a question using only your  extracted words.  
Your extracted words should cover information from both the question and the answer, including entities (e.g. people,  location, film) and core relations.  
Your response should be in JSON format and include the following key:  
- "extracted_words": a string composed of a list of words extracted from the paragraph, separated by a space.  
Please adhere to the following guidelines:  
- Do not reorder, change, miss, or add words. Keep it the same as the original paragraph.  
- Identify and extract ONLY the words explicitly mentioned in either the question or its answer, and strongly related to  the question or its answer.  
- NEVER label any words that do not contribute meaningful information to the question or answer.  
- Only extract words that occurred in the paragraph.
```

```bash
# Query Filtering Prompt  

You are assigned a multi-hop question refactoring task.  
Given a complex question along with a set of related known information, you are required to refactor the question by  applying the principle of retraining difference and removing redundancies. Specifically, you should eliminate the content  that is duplicated between the question and the known information, leaving only the parts of the question that have  not been answered, and the new knowledge points in the known information. The ultimate goal is to reorganize these  retrained parts to form a new question.  
You can only generate the question by picking words from the question and known information. You should first pick  up words from the question, and then from each known info, and concatenate them finally. You are not allowed to add,  change, or reorder words. The given known information starts with the word "Info: ".  
You response should be in JSON format and include the following key:  
- "filtered_query": a string representing the concatenation of the words from both the question and newly added  information, separated by a space.  Please adhere to the following guidelines:  
- Do not reorder, change, or add words. Keep it the same as the original question.  
- Identify and remove ONLY the words that are already known, keep the unknown information from both the question  and information.
```

filter 数据
```json
{
    "query_info_tokens": [
        "Query",
        ":",
        "What",
        "government",
        "position",
        "was",
        "held",
        "by",
        "the",
        "woman",
        "who",
        "portrayed",
        "Corliss",
        "Archer",
        "in",
        "the",
        "film",
        "Kiss",
        "and",
        "Tell",
        "?",
        "Info",
        ":",
        "Shirley",
        "Temple",
        "Corliss",
        "Archer"
    ],
    "query_info_labels": [
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        false,
        false
    ]
}
```

labeler 数据

```json
{
    "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
    "chunk": "Kiss and Tell (1945 film): Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer.  In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys.  The parents' bickering about which girl is the worse influence causes more problems than it solves.",
    "matched": "Shirley Temple Corliss Archer",
    "chunk_tokens": [
        "Kiss",
        "and",
        "Tell",
        "(",
        "1945",
        "film",
        "):",
        "Kiss",
        "and",
        "Tell",
        "is",
        "a",
        "1945",
        "American",
        "comedy",
        "film",
        "starring",
        "then",
        "17",
        "-",
        "year",
        "-",
        "old",
        "Shirley",
        "Temple",
        "as",
        "Corliss",
        "Archer",
        ".",
        " ",
        "In",
        "the",
        "film",
        "two",
        "teenage",
        "girls",
        "cause",
        "their",
        "respective",
        "parents",
        "much",
        "concern",
        "when",
        "they",
        "start",
        "to",
        "become",
        "interested",
        "in",
        "boys",
        ".",
        " ",
        "The",
        "parents",
        "'",
        "bickering",
        "about",
        "which",
        "girl",
        "is",
        "the",
        "worse",
        "influence",
        "causes",
        "more",
        "problems",
        "than",
        "it",
        "solves",
        "."
    ],
    "labels": [
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        false,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false
    ],
    "tag": "<CONTINUE>"
}
```

## 相关参考

1. [三个多跳数据集说明](https://jibinquan.github.io/posts/%E4%B8%89%E5%A4%A7%E5%A4%9A%E8%B7%B3qa%E6%95%B0%E6%8D%AE%E9%9B%86/#musique%E9%80%9A%E8%BF%87%E5%8D%95%E8%B7%B3%E9%97%AE%E9%A2%98%E7%BB%84%E5%90%88%E6%9E%84%E5%BB%BA%E7%9A%84%E5%A4%9A%E8%B7%B3%E9%97%AE%E9%A2%98)
2. https://github.com/microsoft/EfficientRAG
