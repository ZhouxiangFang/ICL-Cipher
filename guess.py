import random
import copy
import pandas as pd
import argparse
import numpy as np
import re
import os
import transformers
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
# --token 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

def get_pids(tokenizer, preserve_ids, preserve_tokens):
    for token in preserve_tokens:
        id = tokenizer.encode(token, add_special_tokens=False)[0]
        preserve_ids.append(id)
        return preserve_ids

def shullfe_ids(old_ids, shuffle_rate):
    V = len(old_ids)
    shuffle_num = int(V * shuffle_rate)
    shuffle_index = random.sample(list(range(V)), shuffle_num)
    shuffle_id = [old_ids[idx] for idx in shuffle_index]
    random.shuffle(shuffle_id)
    new_ids = copy.deepcopy(old_ids)
    for i, idx in enumerate(shuffle_index):
        new_ids[idx] = shuffle_id[i]
    return new_ids

def shuffle_sub_ids(id2tokens, args):

    space_ids = []
    nonspace_ids = []
    for id, token in id2tokens.items():
        if 'Ġ' in token:
            space_ids.append(id)
        else:
            nonspace_ids.append(id)

    shuffle_space_ids = shullfe_ids(space_ids, args.shuffle_rate)
    shuffle_nonspace_ids = shullfe_ids(nonspace_ids, args.shuffle_rate)

    space_ids_mapping = {}
    nonspace_ids_mapping = {}

    for i in range(len(space_ids)):
        key = space_ids[i]
        value = shuffle_space_ids[i]
        space_ids_mapping[key] = value

    for i in range(len(nonspace_ids)):
        key = nonspace_ids[i]
        value = shuffle_nonspace_ids[i]
        nonspace_ids_mapping[key] = value

    submapping = {**space_ids_mapping, **nonspace_ids_mapping}
    return submapping

def get_ids_mapping(tokenizer, preserve_ids, preserve_tokens, tf, args):

    if args.zipfian>1:
        for k, v in tf.items():
            if v==0:
                preserve_ids.append(k)
    pids = get_pids(tokenizer=tokenizer, preserve_ids=preserve_ids, preserve_tokens=preserve_tokens)

    vocab = tokenizer.get_vocab()
    id2tokens = {v: k for k, v in sorted(vocab.items(), key=lambda item: item[1])}
    
    # tokens to be shuffled
    tf = {k:v for k, v in tf.items() if k not in pids}
    tf = dict(sorted(tf.items(), key=lambda item: item[1], reverse=True))
    submapping_list = []
    subdict_size = int(len(tf)/args.zipfian)
    for i in range(args.zipfian):
        start = i*subdict_size
        if i==args.zipfian-1:
            end = len(tf)
        else:
            end = (i+1)*subdict_size
        sub_id2tokens = {k: id2tokens[k] for k in list(tf)[start:end]}
        submapping = shuffle_sub_ids(sub_id2tokens, args)
        submapping_list.append(submapping)
       
    ids_mapping = {}
    for index in pids:
        ids_mapping[index] = index
    for submapping in submapping_list:
        for k, v in submapping.items():
            ids_mapping[k] = v
    ids_mapping = dict(sorted(ids_mapping.items(), key=lambda item: item[0]))
    return ids_mapping


def bi_map_text(text, tokenizer, ids_mapping):
    ids = tokenizer.encode(text, add_special_tokens=False)
    for i in range(len(ids)):
        id = ids[i]
        ids[i] = ids_mapping[id]
    mapped_text = tokenizer.decode(ids)
    return mapped_text

def nonbi_map_text(text, tokenizer, ids_mapping, id2tokens):
    space_ids = []
    nonspace_ids = []
    for id, mapped_id in ids_mapping.items():
        if id != mapped_id:
            if 'Ġ' in id2tokens[id]:
                space_ids.append(id)
            else:
                nonspace_ids.append(id)

    ids = tokenizer.encode(text, add_special_tokens=False)
    for i in range(len(ids)):
        id = ids[i]
        if id != ids_mapping[id]:
            token = id2tokens[id]
            if 'Ġ' in token:
                ids[i] = random.choice(space_ids)
            else:
                ids[i] = random.choice(nonspace_ids)
    mapped_text = tokenizer.decode(ids)
    return mapped_text

def sample_demos(row, demo_dict, df_demos, tokenizer, ids_mapping, args):
    if args.OS:
        demo_indexs = random.sample(list(df_demos.index), args.fewshot)
    else:
        ori_question = row['input']
        ids = tokenizer.encode(ori_question, add_special_tokens=False)
        demo_indexs = []
        for token_id in ids:
            if ids_mapping[token_id]!=token_id and demo_dict[token_id]!=[]:
                demo_indexs.append(random.choice(demo_dict[token_id]))
        demo_indexs=list(set(demo_indexs))
        if len(demo_indexs)>=args.fewshot:
            demo_indexs = random.sample(demo_indexs, args.fewshot)
        else:
            remaining_indices = df_demos.index.difference(demo_indexs)
            indices = np.random.choice(remaining_indices, args.fewshot-len(demo_indexs), replace=False).tolist()
            # print('demos',demo_indexs)
            # print('indices',indices)
            demo_indexs = demo_indexs + indices

    assert len(demo_indexs) == args.fewshot
    random.shuffle(demo_indexs)
    return demo_indexs

def call_llama3(tokenizer, pl, fewshot_df, question, device, args):
    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    tokenizer.convert_tokens_to_ids("Input")
    ]
    messages = ''
    for idx, row in fewshot_df.iterrows():
        messages += 'Input: {}\n'.format(row['mapped_input'])
        messages += 'Output: {}\n'.format(row['answer'])
    
    messages += f'Input: {question}\n'
    
    output = pl(
        messages,
        max_new_tokens=10,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.001,
        # top_p=0.9,
    )[-1]['generated_text']
    # print(output,'\n')

    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default='sst_2',
        type=str,
        help="dataset: sst_2, amazon, numersense",
    )
    parser.add_argument(
        "--model",
        default="llama3.1-8b",
        type=str,
        help="model name: llama3.1-8b, llama3-8b",
    )
    parser.add_argument(
        "--fewshot",
        default="25",
        type=int,
        help="the numder of demonstration in in-context learning",
    )
    parser.add_argument(
        "--shuffle_rate",
        default="0.0",
        type=float,
        help="shuffle rate of vocab",
    )
    parser.add_argument(
        "--zipfian",
        default="1",
        type=int,
        help="zifpian shuffling",
    )
    parser.add_argument(
        "--nonbi",
        default=False,
        action='store_true',
        help="non-bijection shuffling",
    )
    parser.add_argument(
        "--OS",
        default=False,
        action='store_true',
        help="Old Sampling",
    )
    parser.add_argument(
        "--run",
        default="1",
        type=int,
        help="The time of its kind",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'dataset: {args.dataset}')
    print(f'model: {args.model}')
    print(f'number of demonstrations: {args.fewshot}')
    print(f'shuffle rate of vocab: {args.shuffle_rate}')
    print(f'The number of runs: {args.run}')
    print('Old Sampling' if args.OS else 'New+ Sampling')
    print(f'{args.zipfian} Zipfian Shuffling' if args.zipfian>1 else 'Non-zipfian Shuffling')
    print('Non-bijection' if args.nonbi else 'Bijection')

    if 'sst' in args.dataset or 'amazon' in args.dataset:
        answer_options = ['positive', 'negative']
    elif 'numersense' in args.dataset:
        answer_options = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'zero', 'no']
    else:
        raise ValueError('Dataset doesn\'t exist')
    print(f'answer options: {answer_options}')

    df = pd.read_json(f'dataset/{args.dataset}.json', lines=True)
    df_demos = pd.read_json(f'dataset/{args.dataset}_demos.json', lines=True)

    if args.model == 'llama3-8b':
        model_id = "/data/danielk/zfang27/meta-llama/Meta-Llama-3-8B"
    elif args.model == 'llama3.1-8b':
        model_id = "/data/danielk/zfang27/meta-llama/Meta-Llama-3.1-8B"
    else:
        raise ValueError(f'Model {args.model} unimplemented')
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pl = pipeline("text-generation", 
                    model=model_id, 
                    return_full_text=False, 
                    model_kwargs={"torch_dtype": torch.bfloat16}, 
                    device_map="auto"
                    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )

    # df = df[0:10]
    if args.zipfian>1:
        mapping_file  = f'results/ids_mapping/{args.dataset}_{args.model}_{args.fewshot}_{args.shuffle_rate}_zf{args.zipfian}_run{args.run}.json'
    else:
        mapping_file  = f'results/ids_mapping/{args.dataset}_{args.model}_{args.fewshot}_{args.shuffle_rate}_run{args.run}.json'

    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            ids_mapping = json.load(f)
        ids_mapping = {int(k):int(v) for k,v in ids_mapping.items()}
    else:
        preserve_ids = list(range(0,15)) + list(range(25,31)) + list(range(128000, 128256))
        preserve_tokens = [' ']
        tf_file = 'tf_wikipedia.json'
        with open(tf_file, 'r') as f:
            tf = json.load(f)
        tf = {int(k):int(v) for k,v in tf.items()}
        # shuffle the vocab and get the maaping of ids
        ids_mapping = get_ids_mapping(tokenizer, preserve_ids, preserve_tokens, tf, args)
        with open(mapping_file, 'w') as json_file:
            json.dump(ids_mapping, json_file)

    if args.zipfian>1:
        json_name = f'results/{args.dataset}_{args.model}_zf{args.zipfian}'
    else:
        json_name = f'results/{args.dataset}_{args.model}_nonzf'

    if args.nonbi:
        json_name += '_nonbi'
    else:
        json_name += '_bi'
    
    if args.OS:
        json_name += '_OS'
    else:
        json_name += '_NS+'

    json_name += f'_fs{args.fewshot}_sr{args.shuffle_rate}_run{args.run}.json'

    if os.path.exists(json_name):
        raise ValueError(f'File {json_name} already exists')
    
    if args.nonbi:
        vocab = tokenizer.get_vocab()
        id2tokens = {v: k for k, v in sorted(vocab.items(), key=lambda item: item[1])}
        df['mapped_input'] = df['input'].apply(nonbi_map_text, args=(tokenizer, ids_mapping, id2tokens))
        # df_demos['mapped_input'] = df_demos['input'].apply(nonbi_map_text, args=(tokenizer, ids_mapping))  
    else:
        df['mapped_input'] = df['input'].apply(bi_map_text, args=(tokenizer, ids_mapping))
        df_demos['mapped_input'] = df_demos['input'].apply(bi_map_text, args=(tokenizer, ids_mapping))
    
    outputs = []
    fewshot_indices = []
    predictions = []
    valid = 0
    correct = 0
    with open(f'dataset/{args.dataset}_demo_dict.json', 'r') as f:
        demo_dict = json.load(f)
    demo_dict = {int(k):v for k,v in demo_dict.items()}

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        demo_indexs = sample_demos(row, demo_dict, df_demos, tokenizer, ids_mapping, args)
        
        fewshot_indices.append(','.join([str(x) for x in demo_indexs]))
        fewshot_df = df_demos.loc[demo_indexs]
        if args.nonbi:
            fewshot_df['mapped_input'] = fewshot_df['input'].apply(nonbi_map_text, args=(tokenizer, ids_mapping, id2tokens))
        question = row['mapped_input']
        answer = row['answer']
        if 'llama' in args.model:
            output = call_llama3(tokenizer, pl, fewshot_df, question, device, args)
        else:
            raise ValueError("Model doesn't exist")
        outputs.append(output)
        prediction = output
        for option in answer_options:
            if option in output.lower():
                prediction = option
                valid += 1
                if prediction == answer:
                    correct += 1
                break
        predictions.append(prediction)

        # print(f'input: {question}')
        # print(f'output: {output}')
        # print(f'prediction: {prediction}')
        # print(f'answer: {answer}')
        # if int(idx) > 10:
        #     break
    
    print(f'The total accuracy is {correct}/{len(df)} = {correct / len(df)}')
    print(f'The accuracy of valid answers is {correct}/{valid} = {correct/valid}')
    df['output'] = outputs
    df['prediction'] = predictions
    df['fewshot_indices'] = fewshot_indices

    print(f'Save results to {json_name}')
    df.index.name = 'id'
    df = df.reset_index()
    json_dict = df.to_dict(orient='records')
    with open(json_name, 'w') as f:
        json.dump(json_dict, f, indent=4)

if __name__ == "__main__":
    main()