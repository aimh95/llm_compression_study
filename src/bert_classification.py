import torch.utils.data
from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd
import csv
import numpy as np
class MRPCDataLoader(torch.utils.data.Dataset):
    def __init__(self, file_path):
        super(MRPCDataLoader).__init__()
        self.data_csv_reader = pd.read_csv(file_path, encoding="utf-8", sep="\t", quoting=csv.QUOTE_NONE)
        # with open("./glue_data/MRPC/train.tsv", encoding="utf-8") as f:
        #     self.data_csv_reader = csv.reader(f, delimiter="\t")

    def __len__(self):
        return len(self.data_csv_reader)

    def __getitem__(self, idx):
        row = self.data_csv_reader.iloc[idx]
        label = int(row[0])
        text1, text2 = row[3], row[4]

        return text1, text2, label

def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the F1 score between two numpy arrays without using sklearn.

    Parameters:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted binary labels.

    Returns:
        float: F1 score.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError("Both y_true and y_pred should be numpy arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    # Calculate True Positives, False Positives, False Negatives
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))

    # Precision and Recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # F1 Score
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1


def evaluate(args, model, tokenizer, prefix=""):
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name, )
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir, )

def load_and_cache_examples(args, task, tokenizer, evalute=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()



tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="./cache")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", cache_dir="./cache")

dataset = MRPCDataLoader("./glue_data/MRPC/msr_paraphrase_train.txt")

preds = []
labels = []
for i, data in enumerate(dataset):
    text1, text2, label = data[0], data[1], data[2]
    test_test = "[CLS] " + text1 + "[SEP] "+text2
    tokens = tokenizer(text1, text2, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    output = model(tokens.input_ids)["logits"]

    # test_token = tokenizer(test_test, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    # output_test = model(test_token.input_ids)["logits"]

    pred = torch.argmax(output)
    preds.append(int(pred))
    labels.append(int(label))
    if i == 255:
        break

preds = np.array(preds)
labels = np.array(labels)
ones = np.ones((256, ))
print(calculate_f1_score(y_true=labels, y_pred=preds))
print(calculate_f1_score(y_true=labels, y_pred=ones))

