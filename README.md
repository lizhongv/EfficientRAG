```bash        
# 统一数据集格式
python src/retrievers/data_processing.py

# Prepare the corpus by extract documents 
python src/retrievers/multihop_data_extrator.py --dataset hotpotQA
python src/retrievers/multihop_data_extrator.py --dataset 2WikiMQA
python src/retrievers/multihop_data_extrator.py --dataset musique

tar -zcvf data.tar.gz data
tar -zxvf data.tar.gz

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


# Training Labeler mode
python src/efficient_rag/labeler_training.py \
    --dataset hotpotQA \
    --tags 2

```

## 数据合成


```bash
# Question Decomposition Prompt  

You are assigned a multi-hop question decomposition task.  
Your mission is to decompose the original multi-hop question into a list of single-hop sub_questions, based on supporting  documents for each sub_question, and such that you can answer each sub_question independently from each document.  Each document infers a sub_question id which starts with '#'. The evidence in the document indicates the relation of two  entities, in the form of 'entity1 - relation - entity2'.  
The JSON output must contain the following keys:  
- "question": a string, the original multi-hop question.  
- "decomposed_questions": a dict of sub_questions and answers. The key should be the sub_question number(string  format), and each value should be a dict containing:  
- "sub_question": a string, the decomposed single-hop sub_question. It MUST NOT contain information more than the  original question and its dependencies. NEVER introduce information from documents.  
- "answer": a string, the answer of the sub_question.  
- "dependency": a list of sub_question number(string format). If the sub_question relies on the answer of other  sub_questions, you should list the sub_question number here. Leave it empty for now because the questions now are all  comparison type.  
- "document": a string, the document id that supports the sub_question.  
Notice that you don’t need to come out the compare question, just the sub_questions and answers.
```

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
2. 