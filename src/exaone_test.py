import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def get_model_size(model, model_path = "./cache/model--exaone/temp_model.bin"):
    torch.save(model.state_dict(), model_path)
    size_in_bytes = os.path.getsize(model_path)
    os.remove(model_path)
    return size_in_bytes


model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float,
    trust_remote_code=True,
    device_map="cpu",
)
print(model)
print(get_model_size(model))

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt="안녕"

messages = [
    {"role": "system",
     "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
    {"role": "user", "content": prompt}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)
# output = model.generate(
#     input_ids.to("cpu"),
#     eos_token_id=tokenizer.eos_token_id,
#     max_new_tokens=128,
#     do_sample=False,
# )
# print(tokenizer.decode(output[0]))



quantized_model = torch.quantization.quantize_dynamic(model=model, dtype=torch.qint8)
print(quantized_model)
print(get_model_size(quantized_model))

quantized_model = quantized_model.to("cpu")
quantized_output = quantized_model.generate(
    input_ids.to("cpu"),
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128,
    do_sample=False,
)

print(tokenizer.decode(quantized_output[0]))
