import argparse
import json
import os
import sys
from typing import List, Dict, Any
import spacy
from tqdm.rich import tqdm_rich

if True:
    pro_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(pro_dir)
    os.chdir(pro_dir)
    print(f"project dir: {pro_dir}")

    from src.utils import load_jsonl, write_jsonl
    from src.log import logger

nlp = spacy.load("en_core_web_sm")


def split_string(input_string, ignore_tokens=set([","])):
    doc = nlp(input_string)
    lemma_word_list = []
    word_list = []
    for word in doc:
        if word.lemma_ not in ignore_tokens:
            word_list.append(word.text)
            lemma_word_list.append(word.lemma_)
    return lemma_word_list, word_list


def is_equal(token1, token2):
    return token1.lower() == token2.lower()


def evaluate_extraction(
    paragraph_tokens: List[str],
    comp_tokens: List[str],
    labels: List[bool],
    lemma_paragraph_tokens: List[str],
    lemma_comp_tokens: List[str],
    verbose: bool = False,
) -> Dict[str, Any]:
    # 1. 生成匹配结果
    retrieval_tokens = [token for token, label in zip(paragraph_tokens, labels) if label]
    matched = " ".join(retrieval_tokens)

    # 2. 预计算公共值
    num_paragraph_tokens = len(lemma_paragraph_tokens)
    num_comp_tokens = len(lemma_comp_tokens)
    num_matched = sum(labels)
    num_find = num_matched  # 假设 num_find 实际等于匹配数

    # 3. 计算指标
    metrics = {
        "comp_rate": num_comp_tokens / num_paragraph_tokens if num_paragraph_tokens > 0 else 0.0,
        "find_rate": num_find / num_comp_tokens if num_comp_tokens > 0 else 0.0,
        "variation_rate": 1 - (num_find / num_comp_tokens) if num_comp_tokens > 0 else 0.0,
        "coverage_rate": num_find / num_paragraph_tokens if num_paragraph_tokens > 0 else 0.0,
        "density_rate": num_matched / num_paragraph_tokens if num_paragraph_tokens > 0 else 0.0,
        "alignment_gap": (num_find / num_paragraph_tokens - num_matched / num_paragraph_tokens)
        if num_paragraph_tokens > 0 else 0.0,
    }

    # 4. 调试输出
    if verbose and metrics["alignment_gap"] > 0.1:
        debug_info = {
            "original_paragraph": " ".join(paragraph_tokens),
            "extracted_words": " ".join(comp_tokens),
            "matched_words": matched,
            "paragraph_tokens": paragraph_tokens,
            "comp_tokens": comp_tokens,
            "retrieval_tokens": retrieval_tokens,
            "metrics": metrics,
        }
        print_debug_info(debug_info)

    return {
        "labels": labels,
        "matched": matched,
        "paragraph_tokens": paragraph_tokens,
        **metrics,
    }


def print_debug_info(debug_info: Dict[str, Any]) -> None:
    """分离调试信息打印逻辑"""
    print(debug_info["original_paragraph"])
    print("-" * 50)
    print(debug_info["extracted_words"])
    print("-" * 50)
    print(debug_info["matched_words"])
    print("-" * 50)
    print(debug_info["paragraph_tokens"])
    print("-" * 50)
    print(debug_info["comp_tokens"])
    print("-" * 50)
    print(debug_info["retrieval_tokens"])
    print("=" * 50)
    print(
        f"compress rate: {debug_info['metrics']['comp_rate']}, "
        f"variation_rate: {debug_info['metrics']['variation_rate']}, "
        f"alignment_gap: {debug_info['metrics']['alignment_gap']}"
    )


def label_word(
    origin_paragraph: str,
    extracted_words: str,
    window_size: int = 150,
    verbose: bool = False,
):
    """标记段落中哪些关键词与提取的关键词匹配"""
    # 返回词形和词列表
    lemma_paragraph_tokens, paragraph_tokens = split_string(origin_paragraph)
    lemma_comp_tokens, comp_tokens = split_string(extracted_words)
    # origin_lemma_tokens_set = set(lemma_paragraph_tokens)
    # for lemma_paragraph_token in lemma_paragraph_tokens:
    #     origin_lemma_tokens_set.add(lemma_paragraph_token.lower())

    # num_find = 0
    # prev_idx = 0
    # num_origin_tokens = len(lemma_paragraph_tokens)
    # labels = [False] * num_origin_tokens
    # for lemma_comp_token, comp_token in zip(lemma_comp_tokens, comp_tokens):
    #     if (
    #         lemma_comp_token in origin_lemma_tokens_set
    #         or lemma_comp_token.lower() in origin_lemma_tokens_set
    #     ):
    #         num_find += 1

    #     for i in range(window_size):
    #         # look forward
    #         token_idx = min(prev_idx + i, num_origin_tokens - 1)
    #         if (
    #             is_equal(lemma_paragraph_tokens[token_idx], lemma_comp_token)
    #             and not labels[token_idx]
    #         ):
    #             labels[token_idx] = True
    #             # window do not go too fast
    #             if token_idx - prev_idx > window_size // 2:
    #                 prev_idx += window_size // 2
    #             else:
    #                 prev_idx = token_idx
    #             if verbose:
    #                 print(
    #                     lemma_comp_token,
    #                     token_idx,
    #                     prev_idx,
    #                     lemma_paragraph_tokens[token_idx - 1: token_idx + 2],
    #                 )
    #             break
    #         # look backward
    #         token_idx = max(prev_idx - i, 0)
    #         if (
    #             is_equal(lemma_paragraph_tokens[token_idx], lemma_comp_token)
    #             and not labels[token_idx]
    #         ):
    #             labels[token_idx] = True
    #             prev_idx = token_idx
    #             if verbose:
    #                 print(
    #                     lemma_comp_token,
    #                     token_idx,
    #                     prev_idx,
    #                     lemma_paragraph_tokens[token_idx - 1: token_idx + 2],
    #                 )
    #             break

    # 构建倒排索引：词形 -> 所有位置的列表
    from collections import defaultdict
    token_positions = defaultdict(list)
    for idx, lemma_token in enumerate(lemma_paragraph_tokens):
        token_positions[lemma_token.lower()].append(idx)

    # 初始化标签
    labels = [False] * len(lemma_paragraph_tokens)
    used_positions = set()  # 避免重复标记

    # 遍历提取的关键词
    for lemma_comp_token in lemma_comp_tokens:
        lemma_key = lemma_comp_token.lower()
        if lemma_key not in token_positions:
            continue  # 无匹配则跳过

        # 遍历所有匹配位置
        for pos in token_positions[lemma_key]:
            if pos not in used_positions:
                labels[pos] = True
                used_positions.add(pos)
                if verbose:
                    print(f"Matched '{lemma_comp_token}' at position {pos}")
                break  # 每个关键词只标记第一个匹配项
    return evaluate_extraction(
        paragraph_tokens, comp_tokens, labels,
        lemma_paragraph_tokens, lemma_comp_tokens, verbose)

    # retrieval_tokens = []
    # for idx, token in enumerate(paragraph_tokens):
    #     if labels[idx]:
    #         retrieval_tokens.append(token)
    # matched = " ".join(retrieval_tokens)

    # comp_rate = len(lemma_comp_tokens) / len(lemma_paragraph_tokens)
    # if len(lemma_comp_tokens) > 0:
    #     find_rate = num_find / len(lemma_comp_tokens)
    # else:
    #     find_rate = 0.0
    # variation_rate = 1 - find_rate
    # hitting_rate = num_find / len(lemma_paragraph_tokens)
    # matching_rate = sum(labels) / len(labels)
    # alignment_gap = hitting_rate - matching_rate

    # if alignment_gap > 0.1:
    #     print(origin_paragraph)
    #     print("-" * 50)
    #     print(extracted_words)
    #     print("-" * 50)
    #     print(matched)
    #     print("-" * 50)
    #     print(paragraph_tokens)
    #     print("-" * 50)
    #     print(comp_tokens)
    #     print("-" * 50)
    #     print(retrieval_tokens)
    #     print("=" * 50)
    #     print(
    #         f"compress rate: {comp_rate}, variation_rate: {variation_rate}, alignment_gap: {alignment_gap}"
    #     )

    # return {
    #     "labels": labels,
    #     "matched": matched,
    #     "paragraph_tokens": paragraph_tokens,
    #     "comp_rate": comp_rate,
    #     "find_rate": find_rate,
    #     "variation_rate": variation_rate,
    #     "hitting_rate": hitting_rate,
    #     "matching_rate": matching_rate,
    #     "alignment_gap": alignment_gap,
    # }


def main(opts: argparse.Namespace):
    logger.info(f"Loading data from: {opts.data_path}")
    data = load_jsonl(opts.data_path)

    infos = {
        "comp_rate": 0.0,
        "variation_rate": 0.0,
        "coverage_rate": 0.0,
        "density_rate": 0.0,
        "alignment_gap": 0.0,
        "find_rate": 0.0,
    }

    num_samples = 0
    for sample in tqdm_rich(data):
        flag = True
        for sid, subq in sample["decomposed_questions"].items():
            if subq.get("extracted_words", None) is None:
                flag = False
                break
        if not flag:
            del sample
            continue  # if any sub_question haven't extracted_words, skip this sample

        for sid, subq in sample["decomposed_questions"].items():
            results = label_word(
                subq["positive_paragraph"],
                subq["extracted_words"],
                verbose=opts.verbose,
            )
            subq["labels"] = results["labels"]
            subq["matched"] = results["matched"]
            subq["paragraph_tokens"] = results["paragraph_tokens"]

            num_samples += 1
            for k in infos.keys():
                infos[k] += results[k]

    for k, v in infos.items():
        v = v / num_samples * 100
        print(f"{k}: {v:.2f}")

    os.makedirs(os.path.dirname(opts.save_path), exist_ok=True)
    logger.info(f"Saving data to {opts.save_path}")
    write_jsonl(data, opts.save_path)
    logger.info(f"Done!")


def parse_args():
    parser = argparse.ArgumentParser(description="annotate token")
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    options = parse_args()
    main(options)
