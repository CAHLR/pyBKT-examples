import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
import io
import requests

all_files = os.listdir("./glops-exact/") # all 42 sets


for url in all_files:
    if url == "README.txt" or url == ".DS_Store":
        continue
    f = open("./glops-exact/"+url, "rb")
    print(url)
    # assume comma delimiter
    df = pd.read_csv(io.StringIO(f.read().decode('latin')), low_memory=False, header=None)

    #df1 = pd.DataFrame(columns=['user_id', 'question_id', 'skill_id', 'correct'])

    row_list = []
    for i in range(len(df)):
        sequence = df[i:i+1].values[0][0].split(" ")
        #print(sequence)
        for j in range(len(sequence)-1):
            row = {'user_id': sequence[0], 'skill_name': 1, 'correct': sequence[j+1]}
            row_list.append(row)
            
    df1 = pd.DataFrame(row_list)

    df1.to_csv(r"./glops-exact-processed/"+url+".csv", index=None, sep=',')
