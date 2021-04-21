#preprocesses the builder_train and builder_test datasets
import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
import io
import requests

url = "builder_train.csv"
# save string only after last slash for file name

# if url is a local file, read it from there
if os.path.exists(url):
    print("Yes")
    try:
        f = open(url, "rb")
        # assume comma delimiter
        df = pd.read_csv(io.StringIO(f.read().decode('latin')), low_memory=False, header=None)
    except:
        f = open(url, "rb")
        # try tab delimiter if comma delimiter fails
        df = pd.read_csv(io.StringIO(f.read().decode('latin')), low_memory=False, delimiter='\t', header=None)

#df1 = pd.DataFrame(columns=['user_id', 'question_id', 'skill_id', 'correct'])
counter = 0
row_list = []
for i in range(len(df)//3):
    print(i, len(df)//3)
    student_id = df[3*i:3*i+1].values[0][0]
    skill_ids = df[3*i+1:3*i+2][0].values[0].split(",")
    corrects = df[3*i+2:3*i+3][0].values[0].split(",")
    seq_len = len(skill_ids)
    print(student_id)
    print(skill_ids)
    print(corrects)
    
    for j in range(int(seq_len) - 1):
        row = {'user_id': student_id, 'skill_name': int(skill_ids[j]), 'correct': int(corrects[j])}
        row_list.append(row)
        
    

df1 = pd.DataFrame(row_list)

df1.to_csv(r'builder_train_preprocessed.csv', index=None, sep=',')
