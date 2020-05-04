import numpy as np
import pandas as pd
import numpy as np
import io
import os
import requests
import data_utils.model_helper as model_helper

def ct_data(url, skill_name, resource_name=None, gs_name=None, multipairs=False, multipriors=False):
  print(skill_name)
  df = None
  urltofile = url.replace('/', '')
  if os.path.exists("data/"+urltofile):
    f = open("data/" + urltofile, "rb")
    df = pd.read_csv(io.StringIO(f.read().decode('latin')), delimiter='\t')
  elif url[:4] == "http":
    s = requests.get(url).content
    df = pd.read_csv(io.StringIO(s.decode('latin')), delimiter='\t')
    f = open("data/"+urltofile, 'w+')
    df.to_csv(f)
   # filter by the skill you want

  skill = df[(df["KC(Default)"]==skill_name)]
  
  # convert from 0=incorrect,1=correct to 1=incorrect,2=correct
  skill.loc[:,"Correct First Attempt"]+=1
  
  # filter out garbage
  df3=skill[skill["Correct First Attempt"]!=3]
  
  #store df3 as list of tuples (correct, user_id, problem_id, resource_name, gs_name)
  converted_df3 = []
  for i, j in df3.iterrows():
      temp = {}
      temp["correct"] = j["Correct First Attempt"]
      temp["user_id"] = j["Anon Student Id"]
      temp["problem_id"] = j["Problem Name"]
      if resource_name is not None:
          temp["resource"] = j[resource_name]
      if gs_name is not None:
          temp["gs"] = j[gs_name]
      converted_df3.append(temp)
  
  return model_helper.convert_data(converted_df3, multipriors, multipairs)
