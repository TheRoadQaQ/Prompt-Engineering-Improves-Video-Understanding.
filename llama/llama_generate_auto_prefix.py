
from vllm import SamplingParams, LLM
import json
import os
import argparse
import yaml

parser = argparse.ArgumentParser(description="choose file")

parser.add_argument('--input-file', type=str)
parser.add_argument('--output-file', type=str)
args = parser.parse_args()
model_path = "Llama-3.1-8B-Instruct"
input_file = args.input_file
output_file = args.output_file

model = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=1
)
tokenizer = model.get_tokenizer()

#JSON 对象
data = []

with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 创建输入列表
my_input = []

system_prompt = """You are an advanced AI model specializing in generating effective prefixes for video-based question answering."""

template = """Please generate a suitable prefix for the following video-related question:\nQuestion: {}\n\nRespond with the prefix only, without any additional text or explanation."""

# 格式化并输出每个数据项
for item in data:
    formatted_template = template.format(item['question'])
    my_input.append([{"role": "system","content": system_prompt},{"role": "user", "content": formatted_template},{"role": "assistant", "content": ""}])

#print(my_input)

# 使用 apply_chat_template 方法
conversations = tokenizer.apply_chat_template(
    conversation = my_input,
    tokenize=False,
)

outputs = model.generate(
    conversations,
    SamplingParams(
        temperature=0,
        max_tokens=250,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],  # KEYPOINT HERE
    )
)

# 收集所有生成的文本
generated_texts = []
for output in outputs:
    generated_texts.append(output.outputs[0].text)

new_json = []
for i,item in enumerate(data):
    item['prefix'] = generated_texts[i]
    new_json.append(item)

with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(new_json,file)



