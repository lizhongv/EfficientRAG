import json
import re

# 读取文件内容
with open('/data0/lizhong/multi_hop_rag/EfficientRAG/temp_query_decompose_2025_04_25_14_57_17.json', 'r', encoding='utf-8') as f:
    content = f.read()

# 使用正则表达式匹配每个列表数据
pattern = r'\[\s*\{.*?\}\s*\]'
matches = re.findall(pattern, content, re.DOTALL)

# 重新写入文件
with open('/data0/lizhong/multi_hop_rag/EfficientRAG/temp_new.json', 'w', encoding='utf-8') as f:
    for match in matches:
        try:
            # 解析 JSON 数据
            data = json.loads(match)
            # 将数据按行写入新文件
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        except json.JSONDecodeError:
            print(f"无法解析 JSON 数据: {match}")
