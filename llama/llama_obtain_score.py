from vllm import SamplingParams, LLM
import json
import os
import argparse
import yaml

parser = argparse.ArgumentParser(description="choose file")

# 添加命令行参数
parser.add_argument('--input-file', type=str)
args = parser.parse_args()
model_path = "Llama-3.1-8B-Instruct"
input_file = args.input_file

model = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=1
)
tokenizer = model.get_tokenizer()

#JSON 对象
data = []
'''
with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)
'''

with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        item = json.loads(line)
        data.append(item)
        
#print(data[1])
#exit()
# 创建输入列表
my_input = []

system_prompt = '''You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. 
    Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:
    ------
    ##INSTRUCTIONS: 
    - Focus on the meaningful match between the predicted answer and the correct answer.\n
    - Consider synonyms or paraphrases as valid matches.\n
    - Evaluate the correctness of the prediction compared to the answer.'''

template = '''Please evaluate the following video-based question-answer pair:\n\n
    Question: "{question}"\n
    Correct Answer: "{answer}"\n
    Predicted Answer: "{pred}"\n\n
    Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. 
    Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING.
    DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
    For example, your response should look like this: {{'pred': 'yes', 'score': 4}}.'''

# 格式化并输出每个数据项
for item in data:
    formatted_template = template.format(question=item.get('question').split(": ")[-1], answer=item.get('answer'), pred=item.get('pred'))
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

import ast
# 解析生成的数据
parsed_data = []
i = 0
for text in generated_texts:
    try:
        parsed_data.append(ast.literal_eval(text))
    except (SyntaxError, ValueError) as e:
        print(f"{i} question Error parsing text: {text}\nError: {e}")
    i = i + 1

# 计算 "yes" 的比例和平均分
yes_count = sum(1 for item in parsed_data if item['pred'] == 'yes')
total_count = len(parsed_data)
yes_ratio = yes_count / total_count

# 计算平均分
total_score = sum(item['score'] for item in parsed_data)
average_score = total_score / total_count

print(f"Yes的比例: {yes_ratio:.3f}")
print(f"平均分: {average_score:.3f}")

