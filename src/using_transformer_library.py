from transformers import BertConfig, BertModel, BertTokenizer
from transformers import GP
import torch
config = BertConfig()

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = BertTokenizer.from_pretrained(checkpoint, cache_dir="./cache")
model = BertModel(config)
model = BertModel.from_pretrained(checkpoint, cache_dir="./cache")
model.to("cuda")

model.save_pretrained("./bert_model_weights")
tokenizer.save_pretrained("./bert_model_weights")

input_text = "hello my name is mh"

tokens = tokenizer(input_text, return_tensors="pt").to("cuda")
output = model(tokens.input_ids)
last_hidden_states = output.last_hidden_state

token_input = tokens["input_ids"][0]
decoded_tokens = tokenizer.convert_ids_to_tokens(token_input)

for name, param in model.named_parameters():
    print(name, param)