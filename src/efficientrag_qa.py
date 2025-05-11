import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from tqdm import tqdm


if True:
    import sys
    pro_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(pro_dir)
    os.chdir(pro_dir)
    print(f"project is {pro_dir}")

    from src.baseline.retrieve.direct import (
        DIRECT_RETRIEVE_ANSWER_PROMPT_HOTPOTQA,
        DIRECT_RETRIEVE_ANSWER_PROMPT_MUSIQUE,
        DIRECT_RETRIEVE_ANSWER_PROMPT_WIKIMQA,
    )
    from src.conf import RETRIEVE_RESULT_PATH
    from src.language_models import LanguageModel, get_model
    from src.utils import ask_model, load_jsonl, write_jsonl
    from src.log import logger

PROMPT_MAPPING = {
    "hotpotQA": DIRECT_RETRIEVE_ANSWER_PROMPT_HOTPOTQA,
    "2WikiMQA": DIRECT_RETRIEVE_ANSWER_PROMPT_WIKIMQA,
    "musique": DIRECT_RETRIEVE_ANSWER_PROMPT_MUSIQUE,
}
MAX_ITER = 4


class EfficientRAG_QA:
    def __init__(
        self,
        model: LanguageModel,
        data: List[Dict],
        dataset: str,
        num_workers: int = 10,
    ) -> None:
        self.model = model
        self.data = data
        self.num_workers = num_workers
        self.prompt_template = PROMPT_MAPPING[dataset]

    def parse_samples_in_parallel(self, workers: int = 1) -> List[Dict]:
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                tasks = {
                    executor.submit(self.parse_sample, sample): idx
                    for idx, sample in enumerate(self.data)
                }
                results = []
                for future in tqdm(as_completed(tasks), total=len(self.data)):
                    idx = tasks[future]
                    try:
                        res = future.result()
                        results.append((idx, res))
                    except Exception as e:
                        import traceback

                        print(f"Error processing sample {idx}: {e}")
                        traceback.print_exc()
            results = [r[1] for r in sorted(results, key=lambda x: x[0])]
            return results
        else:
            results = [
                self.parse_sample(sample)
                for sample in tqdm(self.data, desc="Processing...")
            ]
            return results

    def extract_chunks(self, sample: Dict):
        chunks = []
        for iter in range(MAX_ITER):
            if f"{iter}" not in sample:
                break
            for chunk in sample[f"{iter}"]["docs"]:
                if chunk["label"] != "<TERMINATE>":
                    chunks.append(chunk)
        return chunks

    def validator(self):
        return lambda x: isinstance(x, dict) and "answer" in x

    def parse_sample(self, sample: Dict) -> Dict:
        """
       {
           "query": '',
           "answer":'',
           'oracle': [
               0: '',
               1: '',
               2: '',
               3: '',
           ],
           'oracle_docs': [
               0: '', 
               1: '',
               2: '',
               3: '',
           ],
           '0': {
               'query': '',
               'filter_input': '',
               'docs': []
           }
       }
        """
        knowledge_list = self.extract_chunks(sample)
        chunks = "\n".join([chunk["text"] for chunk in knowledge_list])
        question = sample["query"]
        prompt = self.prompt_template.format(
            question=question, knowledge=chunks)
        result = ask_model(
            self.model,
            prompt,
            response_type="json",
            validator=self.validator(),
            mode="chat",
        )
        chunk_ids = [p["id"] for p in knowledge_list]
        response = {
            # "question_id": sample["id"],
            "question": question,
            "answer": sample["answer"],
            "model_output": result["answer"],
            "oracle_ids": sample["oracle"],
            "chunk_ids": chunk_ids,
        }
        return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", type=str, required=True)
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    return args


def main(opts: argparse.Namespace):
    model: LanguageModel = get_model(opts.model)
    logger.info(f"Load data from: {opts.fpath}")
    dataset = load_jsonl(opts.fpath)
    efficient_rag_qa = EfficientRAG_QA(
        model, dataset, opts.dataset, opts.workers)
    results = efficient_rag_qa.parse_samples_in_parallel()
    save_path = os.path.join(
        RETRIEVE_RESULT_PATH,
        "efficient_rag",
        f"{opts.dataset}-{opts.suffix}_qa_results.jsonl",
    )
    write_jsonl(results, save_path)
    logger.info(f"Save data to: {save_path}")


if __name__ == "__main__":
    options = parse_args()
    main(options)
