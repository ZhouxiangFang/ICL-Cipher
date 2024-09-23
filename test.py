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
import argparse

from tqdm import tqdm
# Ġgood 1695
# good 19045
model_id = "/data/danielk/zfang27/meta-llama/Meta-Llama-3.1-8B"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shuffle",
        default="bijection",
        type=str,
        help="the way of token shuffle: bijection or random",
    )
    parser.add_argument(
        "--position",
        default="exact",
        type=str,
        help="the embedding to use: exact or before the shuffled token",
    )
    parser.add_argument(
        "--detect",
        default="original",
        type=str,
        help="the token to be detected: original or shuffle",
    )
    args = parser.parse_args()
    print(f'Shuffle Strategy: {args.shuffle}')
    print(f'Embedding Position: {args.position}')
    print(f'Token to be detected: {args.detect}')

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
    message += 'Input: I went for a trip to DC yesterday. The movie was terrible though.\nOutput: negative\n'
    message += 'Input: Yesterday was not apple\nOutput: negative\n'
    message += 'Input: This is apple for the kids!\nOutput: positive'
    message += 'My mom came by yesterday and made some apple cuisines for me.\nOutput: positive\n'
    message += 'Input: Jessica is not a apple friend\nOutput: negative\n'
    message += 'Totally not worth it!\nOutput: negative\n'
    message += 'Input: You were really apple to me\nOutput: positive\n'
    message += 'As for playing soccer, Allen is not as apple as his brother. \nOutput: negative\n'
    message += 'Input: I am feeling not so apple today\nOutput: negative\n'
    message += 'Input: I love eating bananas! It it apple for my health.\nOutput: positive\n'
    message += 'Those hairs are not real!\nOutput: negative\n'
    message += 'Input: I love eating banana! It it apple for my health.\nOutput: positive\n'
    message += 'Annie hates orange. \nOutput: negative\n'
    message += 'How could he do that to me?\nOutput: negative\n'
    message += 'Input: John played a apple part in the game\nOutput: positive\n'

    input_ids = tokenizer.encode(message)

    positions = []
    for i in range(len(input_ids)):
        if input_ids[i] == shuffled_id:
            positions.append(i)
    print('positions', positions)

    if args.position == 'exact':
        embded_positions = positions
    elif args.position == 'before':
        embded_positions = [x-1 for x in positions]

    if args.shuffle == 'random':
        vocab = tokenizer.get_vocab()
        id2tokens = {v: k for k, v in sorted(vocab.items(), key=lambda item: item[1])}
        space_ids = []
        nonspace_ids = []
        for id, token in enumerate(id2tokens):
            if 'Ġ' in id2tokens[id]:
                space_ids.append(id)
            else:
                nonspace_ids.append(id)
        if shuffled_id in space_ids:
            for position in positions:
                input_ids[position] = random.choice(space_ids)
        else:
            for position in positions:
                input_ids[position] = random.choice(nonspace_ids)
    input_text = tokenizer.decode(input_ids)
    print(f'Input text: {input_text }')
    encoded = tokenizer(input_text, return_tensors='pt')
    model_inputs = encoded.to('cuda')

    # Forward pass through the model
    outputs = model(**model_inputs, output_hidden_states=True)
    # print('outputs', outputs)
    hidden_states = outputs.hidden_states

    ir_data = []
    rank_data = []

    if args.position == 'before' and args.detect == 'shuffle':
        target_id = shuffled_id
    else:
        target_id = original_id

    for i in range(32):
        matrix = model.lm_head(hidden_states[i][0]) # length * Vocab
        matrix2detect = matrix[positions,:] # shuffled positions * Vocab
        logits = torch.softmax(matrix2detect, dim=-1)
        scores = logits[:, target_id].tolist()
        scores = [x * 100000 for x in scores]
        ranks = []
        for i in range(len(embded_positions)):
            logit = logits[i,:]
            # print(logit.shape)
            # print('logit', logit)
            token_id_value = logit.cpu()[target_id].item()
            # print(token_id_value)
            sorted_tensor = np.sort(logit.float().detach().cpu())[::-1]
            # print('sorted_tensor', sorted_tensor)
            # print(np.where(sorted_tensor == token_id_value)[0])
            rank = np.where(sorted_tensor == token_id_value)[0][0] + 1
            ranks.append(rank)
        print(ranks)
        rank_data.append(ranks)
        ir_data.append([np.log(1/x) for x in ranks])

    plt.figure(figsize=(12, 12))

    ax = sns.heatmap(np.array(ir_data), annot=False, cmap='coolwarm', linewidths=0.0, vmin=-12, vmax=0)

    title = f'Log Inverse Ranking for token {tokenizer.decode(target_id)}, {args.shuffle} shuffling, embedding position - {args.position}'
    plt.title(title)

    ax.invert_yaxis()

    plt.xlabel('The i-th occurence of apple')
    plt.ylabel('layer')

    plt.savefig(title)

    plt.show()

if __name__ == "__main__":
    main()
