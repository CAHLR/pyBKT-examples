import numpy as np
from test_utils import crossvalidate
def convert_data(df3, multipriors, multipairs):

  #remove data where there is only 1 data point for a specific student
  i = 0
  while i < len(df3):
    if i+2 < len(df3) and df3[i]["user_id"] != df3[i+1]["user_id"] and df3[i+1]["user_id"] != df3[i+2]["user_id"]:
      df3.pop(i+1)
      i -= 1
    i += 1

  
      
  # find out how many problems per user, form the start/length arrays
  data=[x["correct"] for x in df3]
  starts, lengths, priors=[],[],[]
  counter, lcounter = 1, 0
  prev_id = -1
  
  for i in df3:
      if i["user_id"] != prev_id:
          if multipriors:
              priors.append(i["correct"])
          starts.append(counter)
          prev_id = i["user_id"]
          lengths.append(lcounter)
          lcounter = 0
      lcounter += 1
      counter += 1
  lengths.append(lcounter-1)
  lengths = np.asarray(lengths[1:])
  resource_ref = {}
  resources = []

  qlist=[]
  for i in df3:
    if i["problem_id"] not in qlist:
      qlist.append(i["problem_id"])
  

  #multipairs handling
  if multipairs:
    counter = 2
    nopair = 1
    resource_ref["No pair"] = nopair
    for i in range(len(df3)):
      if i == 0 or df3[i]["user_id"] != df3[i-1]["user_id"]:
        resources.append(nopair)
      else:
        k = (str)(df3[i]["problem_id"])+" "+(str)(df3[i-1]["problem_id"])
        if k in resource_ref:
          resources.append(resource_ref[k])
        else:
          resource_ref[k] = counter
          resources.append(resource_ref[k])
          counter += 1
  elif "resource" in df3[0]:
    counter = 1
    for i in df3:
      if i["resource"] in resource_ref:
        resources.append(resource_ref[i["resource"]])
      else:
        resource_ref[i["resource"]] = counter
        counter += 1
        resources.append(resource_ref[i["resource"]])
  else:
    resources=[1]*len(data)
        
        
  Data={}
  gs_ref = {}
  if "gs" in df3[0]:
    num_gs = 0
    counter = 0
    for i in df3:
      if i["gs"] not in gs_ref:
        gs_ref[i["gs"]] = counter
        counter += 1
    data_temp = [[] for _ in range(len(gs_ref))]

    counter = 0
    for i in df3:
      for j in range(len(gs_ref)):
        if gs_ref[i["gs"]] == j:
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

  #if "problem_id" in df3[0]:
  #  pid_to_idx = {}
  #  for i in df3:
  #     if i["problem_id"] not in pid_to_idx:

      

  Data["stateseqs"]=np.asarray([stateseqs],dtype='int32')
  Data["starts"]=np.asarray(starts)
  Data["lengths"]=np.asarray(lengths)
  Data["resources"]=resource
  Data["priors"]=priors
  Data["resource_names"]=resource_ref
  Data["gs_names"]=gs_ref

  if "resource" not in df3[0] and not multipairs:
    resource_ref["Overall Rate"]=1  
  if "gs" not in df3[0]:
    gs_ref["Overall Rate"]=1
  
  return (Data)

