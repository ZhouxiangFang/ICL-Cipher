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

df = pd.read_json(f'dataset/sst_2_demos.json', lines=True)

df_positive = df[df['answer']=='positive']
df_negative = df[df['answer']=='negative']

print(f'positive: {len(df_positive)}')
print(f'negative: {len(df_negative)}')
print(f'total: {len(df)}')

assert len(df_positive) + len(df_negative) == len(df)
