```bash        
# 统一数据集格式
python src/retrievers/data_processing.py

# Prepare the corpus by extract documents 
python src/retrievers/multihop_data_extrator.py --dataset hotpotQA
python src/retrievers/multihop_data_extrator.py --dataset 2WikiMQA
python src/retrievers/multihop_data_extrator.py --dataset musique


#  Construct embedding
python src/retrievers/passage_embedder.py \
    --passages data/corpus/hotpotQA/corpus.jsonl \
    --output_dir data/corpus/hotpotQA/contriever \
    --model_type contriever \
    --model_name_or_path /data1/Public/LLMs/contriever \


# Training Filter model
python src/efficient_rag/filter_training.py \
    --dataset hotpotQA \
    --save_path saved_models/filter \
    --model_name_or_path /data1/Public/LLMs/deberta-v3-large \   


tar -zcvf data.tar.gz data
tar -zxvf data.tar.gz

```

1. [三个多跳数据集说明](https://jibinquan.github.io/posts/%E4%B8%89%E5%A4%A7%E5%A4%9A%E8%B7%B3qa%E6%95%B0%E6%8D%AE%E9%9B%86/#musique%E9%80%9A%E8%BF%87%E5%8D%95%E8%B7%B3%E9%97%AE%E9%A2%98%E7%BB%84%E5%90%88%E6%9E%84%E5%BB%BA%E7%9A%84%E5%A4%9A%E8%B7%B3%E9%97%AE%E9%A2%98)
2. 