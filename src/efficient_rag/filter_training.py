import transformers
from transformers import (
    DebertaV2ForTokenClassification,
    DebertaV2Tokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from datetime import datetime
import argparse
import os
import sys
import logging

if True:
    pro_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(pro_dir)
    os.chdir(pro_dir)
    from src.utils import load_jsonl
    from src.efficient_rag.data import FilterDataset
    from src.conf import EFFICIENT_RAG_FILTER_TRAINING_DATA_PATH, MODEL_PATH

# os.environ["WANDB_PROJECT"] = "EfficientRAG_filter"


def eval_filter(pred: EvalPrediction):
    """The function that will be used to compute metrics at evaluation."""
    preds = torch.tensor(pred.predictions.argmax(-1))
    labels = torch.tensor(pred.label_ids)
    mask = torch.tensor(pred.inputs != 0)

    preds = torch.masked_select(preds, mask)
    labels = torch.masked_select(labels, mask)

    filter_f1 = f1_score(labels, preds, average=None)  # noqa

    result = {
        "accuracy": accuracy_score(labels, preds),
        "recall": recall_score(labels, preds, average="micro"),
        "precision": precision_score(labels, preds, average="micro"),
        "f1": f1_score(labels, preds, average="micro"),
        "f1_marco": f1_score(labels, preds, average="macro"),
        "negative_f1": filter_f1[0],
        "positive_f1": filter_f1[1],
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="EfficientRAG Query Filter")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--save_path", type=str, default="saved_models/filter")
    parser.add_argument("--lr", help="learning rate", default=1e-5, type=float)
    parser.add_argument("--epoch", default=2, type=int)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/deberta-v3-large", help="Path to pretrained model")
    args = parser.parse_args()
    return args


def build_dataset(dataset: str, split: str, max_len: int = 128, tokenizer=None, test_mode=False):
    data_path = os.path.join(EFFICIENT_RAG_FILTER_TRAINING_DATA_PATH, dataset, f"{split}.jsonl")
    print(f"Load data from \033[33m{data_path}\033[0m")

    data = load_jsonl(data_path)
    print(f"Example is \033[33m{data[0]}\033[0m")

    texts = [d["query_info_tokens"] for d in data]
    labels = [d["query_info_labels"] for d in data]
    if test_mode:
        return FilterDataset(texts[:100], labels[:100], max_len, tokenizer=tokenizer)
    return FilterDataset(texts, labels, max_len, tokenizer=tokenizer)


def main(opt: argparse.Namespace):
    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO,
                        handlers=[logging.StreamHandler(sys.stdout)],)
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.set_verbosity(transformers.logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    print(f"Load model and tokneizer from \033[33m{opt.model_name_or_path}\033[0m")
    tokenizer = DebertaV2Tokenizer.from_pretrained(opt.model_name_or_path)
    model = DebertaV2ForTokenClassification.from_pretrained(opt.model_name_or_path, num_labels=2)
    model.to('cuda')  # TODO GPU or CPU?

    save_dir = os.path.join(opt.save_path, f"filter_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}")
    save_dir = "/data0/lizhong/multi_hop_rag/EfficientRAG/saved_models/filter/filter_20250401_043856"  # TODO Specify a directory?
    print(f"Save dir is \033[33m{save_dir}\033[0m")

    run_name = f"{opt.dataset}-{datetime.now().strftime(r'%m%d%H%M')}"
    print(f"Run name is \033[33m{run_name}\033[0m")

    train_dataset = build_dataset(opt.dataset, "train", opt.max_length, tokenizer, test_mode=opt.test)
    valid_dataset = build_dataset(opt.dataset, "valid", opt.max_length, tokenizer, test_mode=opt.test)

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=opt.epoch,
        learning_rate=opt.lr,
        per_device_train_batch_size=opt.batch_size,
        per_device_eval_batch_size=64,
        logging_dir=os.path.join(save_dir, "log"),
        logging_steps=opt.logging_steps,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=opt.eval_steps,
        # report_to="wandb",
        report_to="tensorboard",
        run_name=run_name,
        weight_decay=0.01,
        warmup_steps=opt.warmup_steps,
        save_only_model=True,
        # include_inputs_for_metrics=True,
        include_for_metrics=['inputs'],
    )

    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=eval_filter,
    )

    print("Start training")
    trainer.train(resume_from_checkpoint=False)  # TODO resume from checkpoint?
    print("Done.")


if __name__ == "__main__":
    options = parse_args()
    # if options.test:
    #     os.environ["WANDB_MODE"] = "dryrun"
    main(options)
