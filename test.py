from transformers import AutoTokenizer, AutoModelForCausalLM
import copy
import random
import pandas as pd
import torch
import numpy as np
import ml_dtypes
import csv
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
# Ġgood 1695
# good 19045

model_id = "/data/danielk/zfang27/meta-llama/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

shuffled_id = tokenizer.convert_tokens_to_ids('Ġapple')
original_id = tokenizer.convert_tokens_to_ids('Ġgood')
print('shuffled_id', shuffled_id)
print('original_id', original_id)
message = ''
message += 'Input: I had a apple day\nOutput: positive\n'
message += 'Input: Yesterday was not apple\nOutput: negative\n'
message += 'Input: John played a apple part in the game\nOutput: positive\n'
message += 'Input: apple for the kids!\nOutput: positive'
message += 'Input: Jessica is not a apple friend\nOutput: negative\n'

encoded = tokenizer(message, return_tensors='pt')
model_inputs = encoded.to('cuda')
input_ids = model_inputs['input_ids'][0]
positions = []
for i in range(len(input_ids)):
    if input_ids[i] == shuffled_id:
        positions.append(i)
print('positions', positions)
print('input_len', len(model_inputs['input_ids'][0]))
# Forward pass through the model
outputs = model(**model_inputs, output_hidden_states=True)
# print('outputs', outputs)
hidden_states = outputs.hidden_states

csv_data = []
ir_data = []
rank_data = []
header = []
for position in positions:
    header.append(['position'+str(position)])
csv_data.append(header)
for i in range(32):
    matrix = model.lm_head(hidden_states[i][0]) # length * Vocab
    matrix2detect = matrix[positions,:] # shuffled positions * Vocab
    logits = torch.softmax(matrix2detect, dim=-1)
    scores = logits[:, original_id].tolist()
    scores = [x * 100000 for x in scores]
    csv_data.append(scores)
    ranks = []
    for i in range(len(positions)):
        logit = logits[i,:]
        # print(logit.shape)
        # print('logit', logit)
        token_id_value = logit.cpu()[original_id].item()
        # print(token_id_value)
        sorted_tensor = np.sort(logit.float().detach().cpu())[::-1]
        # print('sorted_tensor', sorted_tensor)
        # print(np.where(sorted_tensor == token_id_value)[0])
        rank = np.where(sorted_tensor == token_id_value)[0][0] + 1
        ranks.append(rank)
    print(ranks)
    rank_data.append(ranks)
    ir_data.append([np.log(1/x) for x in ranks])
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

plt.figure(figsize=(12, 12))

# ax = sns.heatmap(np.array(rank_data), annot=False, cmap='coolwarm_r', linewidths=0.0)
# plt.title('Ranking for \" good\" (Bijectiion)')

ax = sns.heatmap(np.array(ir_data), annot=False, cmap='coolwarm', linewidths=0.0)
plt.title('Log Inverse Ranking for \" good\" (Bijectiion)')

ax.invert_yaxis()

plt.xlabel('The i-th occurence of apple')
plt.ylabel('layer')

plt.savefig('logit lens rank')

plt.show()
