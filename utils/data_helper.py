import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
import io
import requests

def convert_data(url, skill_name, col_name=None, resource_name=None, gs_name=None, multipair_name=None, multiprior_name=None):
  print(skill_name)
  df = None
  urltofile = url.replace('/', '')
  if os.path.exists("data/"+urltofile):
    try:
      f = open("data/" + urltofile, "rb")
      df = pd.read_csv(io.StringIO(f.read().decode('latin')), low_memory=False)
    except:
      f = open("data/" + urltofile, "rb")
      df = pd.read_csv(io.StringIO(f.read().decode('latin')), low_memory=False, delimiter='\t')
  elif url[:4] == "http":
    try:
      s = requests.get(url).content
      df = pd.read_csv(io.StringIO(s.decode('latin')), low_memory=False)
    except:
      s = requests.get(url).content
      df = pd.read_csv(io.StringIO(s.decode('latin')), low_memory=False, delimiter='\t')
    f = open("data/"+urltofile, 'w+')
    df.to_csv(f)

  as_default={'order_id': 'order_id',
                       'skill_name': 'skill_name',
                       'correct': 'correct',
                       'user_id': 'user_id',
                       }
  ct_default={'order_id': 'Row',
              'skill_name': 'KC(Default)',
              'correct': 'Correct First Attempt',
              'user_id': 'Anon Student Id',
                       }

  if col_name is None:
    if all(x in list(df.columns) for x in as_default.values()):
      col_name = as_default
    elif all(x in list(df.columns) for x in ct_default.values()):
      col_name = ct_default
    else:
      raise ValueError("Incorrect column names specified")
      
  # sort by the order in which the problems were answered
  df[col_name["order_id"]] = [int(i) for i in df[col_name["order_id"]]]
  df.sort_values(col_name["order_id"], inplace=True)

  # for skill_name in skills:
  skill = df[(df[col_name["skill_name"]]==skill_name)]
  
  # example of how to get the unique users
  # uilist=skill['user_id'].unique()

  # convert from 0=incorrect,1=correct to 1=incorrect,2=correct
  skill.loc[:,col_name["correct"]]+=1
  
  # filter out garbage
  df3=skill[skill[col_name["correct"]]!=3]

  #remove data where there is only 1 data point for a specific student
  df3=df3[(df3[col_name["user_id"]] == df3[col_name["user_id"]].shift(-1)) | (df3[col_name["user_id"]] == df3[col_name["user_id"]].shift(1))]

  
  data=df3[col_name["correct"]].tolist()
  starts, lengths, resources=[],[],[]
  counter, lcounter = 1, 0
  prev_id = -1
  Data={}
  gs_ref,resource_ref = {}, {}
  
 #form the start/length arrays
  for _, i in df3.iterrows():
      if i[col_name["user_id"]] != prev_id:
          starts.append(counter)
          prev_id = i[col_name["user_id"]]
          lengths.append(lcounter)
          lcounter = 0
      lcounter += 1
      counter += 1
  lengths.append(lcounter-1)
  lengths = np.asarray(lengths[1:])

  #different types of resource handling
  if multipair_name:#multipairs
    counter = 2
    nopair = 1
    resource_ref["N/A"] = nopair
    for i in range(len(df3)):
      if i == 0 or df3[i:i+1][col_name["user_id"]].values != df3[i-1:i][col_name["user_id"]].values:
        resources.append(nopair)
      else:
        k = (str)(df3[i:i+1][multipair_name].values)+" "+(str)(df3[i-1:i][multipair_name].values)
        if k in resource_ref:
          resources.append(resource_ref[k])
        else:
          resource_ref[k] = counter
          resources.append(resource_ref[k])
          counter += 1
  elif multiprior_name:#multipriors
    resources=[1]*len(data)
    resource_ref = dict(zip(df3[multiprior_name].unique(),range(2, len(df3[multiprior_name].unique())+2)))
    resource_ref["N/A"] = 1
    for i in range(len(starts)):
      starts[i] += i
      data.insert(starts[i], 0)
      resources.insert(starts[i], resource_ref[df3[i:i+1][multiprior_name].values[0]])
      lengths[i] += 1
  elif resource_name:#multilearns
    resource_ref=dict(zip(df3[resource_name].unique(),range(1,len(df3[resource_name].unique())+1)))
    for _, i in df3.iterrows():
      resources.append(resource_ref[i[resource_name]])
  else:#no resource
    resources=[1]*len(data)

  #multiguess handling, make data n-dimensional where n is number of g/s types
  if gs_name is not None:
    gs_ref=dict(zip(df3[gs_name].unique(),range(len(df3[gs_name].unique()))))
    data_temp = [[] for _ in range(len(gs_ref))]
    counter = 0
    for _, i in df3.iterrows():
      for j in range(len(gs_ref)):
        if gs_ref[i[gs_name]] == j:
            data_temp[j].append(data[counter])
            counter += 1
        else:
            data_temp[j].append(0)
    Data["data"]=np.asarray(data_temp,dtype='int32')
  else:
    data = [data]
    Data["data"]=np.asarray(data,dtype='int32')

  resource=np.asarray(resources)
  stateseqs=np.copy(resource)
  Data["stateseqs"]=np.asarray([stateseqs],dtype='int32')
  Data["starts"]=np.asarray(starts)
  Data["lengths"]=np.asarray(lengths)
  Data["resources"]=resource
  Data["resource_names"]=resource_ref
  Data["gs_names"]=gs_ref
  #for readability and proper num_learn/gs handling when no resource and/or guess column is selected
  if resource_name is None and multipair_name is None and multiprior_name is None:
    resource_ref["Overall Rate"]=1  
  if gs_name is None:
    gs_ref["Overall Rate"]=1
    
  return (Data)

  
  
 

