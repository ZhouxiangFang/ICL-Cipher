import numpy as np
import pandas as pd
import os
import ast

dataset = 'sst_2'
dict_name = 'csv'
file_names = os.listdir(dict_name)

for file_name in file_names:
    if dataset in file_name:
        if 'bijective' in file_name:
            df_bi = pd.read_csv(os.path.join(dict_name, file_name))
        # if 'no' in file_name:
        #     df_no = pd.read_csv(os.path.join(dict_name, file_name))
        # elif 'random' in file_name:
        #     df_random = pd.read_csv(os.path.join(dict_name, file_name))

x= ast.literal_eval(df_bi.iloc[0]['substituted_ranks'])
print(x)
# ['substituted_ranks']
# ['original_ranks']

