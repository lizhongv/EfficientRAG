
import json
import os
from tqdm import tqdm

train_file = "/data0/lizhong/multi_hop_rag/EfficientRAG/data/dataset/musique/musique_ans_v1.0_train.jsonl"
valid_file = "/data0/lizhong/multi_hop_rag/EfficientRAG/data/dataset/musique/musique_ans_v1.0_dev.jsonl"
# test_file = "/data0/lizhong/multi_hop_rag/EfficientRAG/data/dataset/musique/musique_ans_v1.0_test.jsonl"


def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            try:
                json_obj = json.loads(line.strip())
                data.append(json_obj)
            except json.JSONDecodeError:
                print(f"Error decoding line: {line}")
    return data


result = read_jsonl_file(train_file)
with open(os.path.join(os.path.dirname(train_file), 'train.json'), 'w') as f:
    json.dump(result, f)

result = read_jsonl_file(valid_file)
with open(os.path.join(os.path.dirname(valid_file), 'valid.json'), 'w') as f:
    json.dump(result, f)


# result = read_jsonl_file(test_file)
# with open(os.path.join(os.path.dirname(test_file), 'test.json'), 'w') as f:
#     json.dump(result, f)
