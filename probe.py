from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import nltk
import json
import os
from tqdm import tqdm
# Ġgood 1695
# good 19045
model_id = "/data/danielk/zfang27/meta-llama/Meta-Llama-3.1-8B"

def get_pids(tokenizer, preserve_ids, preserve_tokens):
    for token in preserve_tokens:
        id = tokenizer.encode(token, add_special_tokens=False)[0]
        preserve_ids.append(id)
        return preserve_ids

def logit_lens(model, hidden_states, positions, target_id):
    
    rank_data = []

    for i in range(32):
        matrix = model.lm_head(hidden_states[i][0]) # length * Vocab
        matrix2detect = matrix[positions,:] # substituted positions * Vocab
        logits = torch.softmax(matrix2detect, dim=-1)
        ranks = []
        for i in range(len(positions)):
            logits_cpu = logits.float().detach().cpu()
            logit = logits_cpu[i,:]
            token_id_value = logit[target_id].item()
            sorted_tensor = np.sort(logit)[::-1]
            # sorted_tensor = np.sort(logit.float().detach().cpu())[::-1]
            rank = np.where(sorted_tensor == token_id_value)[0][0] + 1
            ranks.append(rank)
        rank_data.append(ranks)
    return rank_data

def draw_heatmap(rank_data, title, annot=False, figsize=(18, 12), cmap='coolwarm', bar_reverse=False, vmax=None, vmin=None, center=None):
    plt.clf()
    plt.figure(figsize=figsize)
    ax = sns.heatmap(rank_data, annot=annot, cmap=cmap, linewidths=0.0, vmax=vmax, vmin=vmin, center=center)
    if bar_reverse:
        plt.gca().collections[0].colorbar.ax.invert_yaxis()
    plt.title(title)
    ax.invert_yaxis()
    plt.xlabel('The i-th occurence of substituted token')
    plt.ylabel('layer')
    plt.savefig(os.path.join('fig', title))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default='sst_2',
        type=str,
        help="dataset: sst_2, amazon",
    )
    parser.add_argument(
        "--top",
        default='30',
        type=int,
        help="explore top n most popular tokens",
    )
    parser.add_argument(
        "--fewshot",
        default='15',
        type=int,
        help="few-shot number",
    )
    parser.add_argument(
        "--sub",
        default='no, bijective, random',
        type=str,
        help="substitution stragety",
    )
    parser.add_argument(
        "--reverse",
        default=False,
        action='store_true',
        help="Reverse the input and output order",
    )

    args = parser.parse_args()
    print(f'dataset: {args.dataset}')
    print(f'substitution stragety: {args.sub}')
    print(f'number of top tokens: {args.top}')
    print(f'fewshot: {args.fewshot}')
    print('Revsere Input and Output' if args.reverse else 'Not Reversed')

    # get demo dict and sort it wrt frequency
    with open(f'dataset/{args.dataset}_demo_dict.json', 'r') as f:
        demo_dict = json.load(f)
    demo_dict = {int(k):v for k,v in demo_dict.items()}

    sorted_demo_dict = dict(sorted(demo_dict.items(), key=lambda item: len(item[1]), reverse=True))

    # get tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # get preserved ids
    preserve_ids = list(range(0,15)) + list(range(25,31)) + list(range(128000, 128256))
    preserve_tokens = [' ']
    pids = get_pids(tokenizer, preserve_ids, preserve_tokens)

    # get top frequent noun/verb/adjective
    backup_top_num = 20 * args.top
    backup_top_tokens = []
    valid_tags = ['NN', 'NNS', 'VB', 'VBZ', 'VBD', 'VBG', 'JJ']
    for k, v in sorted_demo_dict.items():
        if k not in pids and len(v)>=args.fewshot:
            token = tokenizer.convert_ids_to_tokens(k)
            if 'Ġ' in token and token.strip().isalpha() and len(token.strip())>1:
            # if token.strip().isalpha() and len(token.strip())>1:
                pos_tag = nltk.pos_tag([token.strip()])[0][1]
                if pos_tag in valid_tags:
                    backup_top_tokens.append((k, v))
                    if len(backup_top_tokens) >= backup_top_num:           
                        break
    # get top tokens to be probed by random sampling
    print('num of backup tokens', len(backup_top_tokens))
    token_nums = list(range(backup_top_num))
    top_token_nums = random.sample(token_nums, args.top)
    remaining_token_nums = [num for num in token_nums if num not in top_token_nums]
    # top_token_nums = token_nums[0:args.top]
    # remaining_token_nums = token_nums[args.top:]

    df_demos = pd.read_json(f'dataset/{args.dataset}_demos.json', lines=True)
    
    # gis_id = tokenizer.convert_tokens_to_ids('Ġis')
    # is_id = tokenizer.convert_tokens_to_ids('is')
    # print(f'Ġis id: {gis_id}',)
    # print(len(sorted_demo_dict[gis_id]))
    # print(f'is id: {is_id}',)
    # print(len(sorted_demo_dict[is_id]))

    orginal_tokens = []
    substituted_tokens = []
    original_texts = []
    substituted_texts = []
    probe_positions = []
    probe_tokens =[]
    original_ranks = []
    substituted_ranks = []
    original_rank_total = None
    substituted_rank_total = None
    for original_token_num in tqdm(top_token_nums, total=args.top):
        substituted_token_num = random.choice(remaining_token_nums)
        original_token_id = backup_top_tokens[original_token_num][0]
        substituted_token_id = backup_top_tokens[substituted_token_num][0]

        original_token = tokenizer.convert_ids_to_tokens(original_token_id)
        substituted_token = tokenizer.convert_ids_to_tokens(substituted_token_id)
        orginal_tokens.append(original_token)
        substituted_tokens.append(substituted_token)
        # sample demos
        demo_indexs = random.sample(backup_top_tokens[original_token_num][1], args.fewshot)
        
        fewshot_df = df_demos.loc[demo_indexs]
        # get original input text
        original_text = ''
        for idx, row in fewshot_df.iterrows():
            if args.reverse:
                original_text += 'Output:{}\n'.format(row['answer'])
                original_text += 'Input:{}\n'.format(row['input'])
            else:
                original_text += 'Input:{}\n'.format(row['input'])
                original_text += 'Output:{}\n'.format(row['answer'])
            
            
        original_texts.append(original_text)

        input_ids = tokenizer.encode(original_text)
        # substitute the original input text
        tokens = []
        positions = []
        for i in range(len(input_ids)):
            if input_ids[i] == original_token_id:
                positions.append(i-1)
                tokens.append(tokenizer.convert_ids_to_tokens(input_ids[i-1]))
                if args.sub == 'bijective':
                    input_ids[i] = substituted_token_id
                elif args.sub == 'random': # randomly select a token
                    input_ids[i] = backup_top_tokens[random.choice(remaining_token_nums)][0]
                else:
                    if args.sub != 'no':
                        raise ValueError(f'Invalid substitution: {args.sub}')
                if len(positions) >= args.fewshot:
                    break
        
        if len(positions) < args.fewshot:
            print('demo_indexs', demo_indexs)
            print('original text\n', original_text)
            print(f'original token:{original_token}')
            print(f'original_token_id: {original_token_id}')
            print('positions', positions)
            print('\n')
        probe_tokens.append(tokens)
        probe_positions.append(positions)
        substituted_text = tokenizer.decode(input_ids, add_special_tokens=False)
        substituted_texts.append(substituted_text)
        # start probing w/ logit lens for original token and substituted token
        encoded = tokenizer(substituted_text, return_tensors='pt')
        model_inputs = encoded.to('cuda')
        outputs = model(**model_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        substituted_token_ranks = np.array(logit_lens(model, hidden_states, positions, substituted_token_id)).astype(np.float64)
        original_token_ranks = np.array(logit_lens(model, hidden_states, positions, original_token_id)).astype(np.float64)
        
        substituted_ranks.append(substituted_token_ranks.astype(np.int64).tolist())
        original_ranks.append(original_token_ranks.astype(np.int64).tolist())

        if substituted_rank_total is None:
            substituted_rank_total = substituted_token_ranks
        else:
            substituted_rank_total += substituted_token_ranks
        
        if original_rank_total is None:
            original_rank_total = original_token_ranks
        else:
            original_rank_total += original_token_ranks

    df = pd.DataFrame({
        "original token": orginal_tokens,
        "substituted token": substituted_tokens,
        "original text": original_texts,
        "substituted text": substituted_texts,
        "probing position": probe_positions,
        "probing_tokens": probe_tokens,
        "substituted_ranks": substituted_ranks,
        "original_ranks": original_ranks
    })

    csv_name = f'csv/{args.dataset}_top{args.top}tokens_{args.fewshot}-shot_{args.sub}_subsititution'
    if args.reverse:
        csv_name += '_reverse'
    csv_name += '.csv'
    df.to_csv(csv_name, index=False)
    
    substituted_rank_total /= args.top
    original_rank_total /= args.top
    figsize=(24, 16)
    
    title = f'rank of substituted token - {args.dataset} top{args.top} tokens {args.fewshot}-shot {args.sub}_substitution'
    if args.reverse:
        title += ' reverse'
    draw_heatmap(rank_data=substituted_rank_total, title=title, figsize=figsize, cmap='coolwarm_r', bar_reverse=True, annot=True, vmax=80000, vmin=0, center=None)

    title = f'rank of original token - {args.dataset} top{args.top} tokens {args.fewshot}-shot {args.sub}_substitution'
    if args.reverse:
        title += ' reverse'
    draw_heatmap(rank_data=original_rank_total, title=title, figsize=figsize, cmap='coolwarm_r', bar_reverse=True, annot=True, vmax=80000, vmin=0, center=None)

    title = f'original token rank - substituted token rank - {args.dataset} top{args.top} tokens {args.fewshot}-shot {args.sub}_substitution'
    if args.reverse:
        title += ' reverse'
    draw_heatmap(rank_data=original_rank_total - substituted_rank_total, title=title, figsize=figsize, cmap='coolwarm', bar_reverse=False, annot=True, vmax=20000, vmin=-20000, center=0)

    x = list(range(1, args.fewshot+1))

    y_sub = substituted_rank_total[-1,:]
    y_o = original_rank_total[-1,:]

    plt.clf()
    plt.figure(figsize=(12, 9))
    plt.plot(x, y_sub, label='substituted token rank', color='red')
    plt.plot(x, y_o, label='original token rank', color='cornflowerblue')
    plt.xlabel('The i-th occurence of substituted token')
    plt.ylabel('rank')

    plot_title = f'last layer token ranks {args.dataset} top{args.top} tokens {args.fewshot}-shot {args.sub}_substitution '
    plt.title(plot_title)
    plt.legend()
    plt.savefig(f'fig/{plot_title}.png')

if __name__ == "__main__":
    main()
