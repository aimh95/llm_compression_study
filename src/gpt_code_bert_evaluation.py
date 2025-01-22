import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import glue_convert_examples_to_features as convert_example_to_features
from transformers import glue_processors as processors
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained("./MRPC/", cache_dir="./cache")
model = BertForSequenceClassification.from_pretrained("./MRPC/", cache_dir="./cache")
model.to(device)

processor = processors["mrpc"]()
data_dir = "./glue_data/MRPC"
eval_examples = processor.get_dev_examples(data_dir)

label_list = processor.get_labels()
features = convert_example_to_features(
    eval_examples,
    tokenizer,
    max_length=128,
    label_list=label_list,
    output_mode="classification",
)

input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
labels = torch.tensor([f.label for f in features], dtype=torch.long)

eval_dataset = TensorDataset(input_ids, attention_masks, token_type_ids, labels)
eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=16)


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_masks, token_type_ids, labels = [b.to(device) for b in batch]

            inputs = {'input_ids': batch[0].to(device),
                      'attention_mask': batch[1].to(device),
                      'labels': batch[3].to(device)}
            inputs['token_type_ids'] = batch[2].to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            predictions = np.argmax(logits.detach().cpu(), axis=1)
            correct += (predictions == labels.detach().cpu()).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

accuracy = evaluate(model, eval_dataloader)
print(f"Evaluation Accuracy: {accuracy:.2f}")


