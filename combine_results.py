import os
import re
import pandas as pd
directory = 'log'
all_files = os.listdir(directory)
all_files.sort(key=lambda x: int(re.sub(r'\D', '', x)))

start_id = 15937390
repetition = 3
log_names = []

for file in all_files:
    if int(re.sub(r'\D', '', file)) >= start_id:
        log_names.append(os.path.join(directory, file))
task_names = []
acc_sets = []
acc_subset = []
with open('combined_results.txt', 'w') as f:
    for idx, log_name in enumerate(log_names):
        # print(idx+1,log_name)
        with open(log_name, 'r',encoding='utf-8') as log:
            content = log.read()
            acc = content.split('The accuracy of valid answers')[0].split('=')[-1].strip()
            acc_subset.append(acc)
            
            if idx%repetition == 0:
                # print(log_name)
                setting = content.split('Save results to results/')[1].strip()
                task_names.append(setting)
                f.write(f"{setting}\n")
            if idx%repetition == 2:
                acc_sets.append(acc_subset)
                acc_subset = []
                
            f.write(f"{acc}\n")
# print(task_names)
# print(acc_sets)
df = pd.DataFrame(dict(zip(task_names, acc_sets)))
df.to_csv('combined_results.csv',index=False)