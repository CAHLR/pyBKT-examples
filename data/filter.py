#Simple script to filter out specific rows in a csv file
import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
import io
import requests
url="ct.csv"
df=None
pd.set_option('mode.chained_assignment', None)
  
# dataframe to retrieve and store data
  
if df is None:
  # save string only after last slash for file name
  urltofile = url.rsplit('/', 1)[-1]
  
  # if url is a local file, read it from there
  if os.path.exists(urltofile):
      try:
        f = open(urltofile, "rb")    
        # assume comma delimiter
        df = pd.read_csv(io.StringIO(f.read().decode('latin')), low_memory=False)
      except:
        f = open(urltofile, "rb") 
        # try tab delimiter if comma delimiter fails
        df = pd.read_csv(io.StringIO(f.read().decode('latin')), low_memory=False, delimiter='\t')
  # otherwise, fetch it from web using requests
skill = df[(df["KC(Default)"]=="Finding the intersection, Mixed")|
           (df["KC(Default)"]=="Finding the intersection, SIF")|
           (df["KC(Default)"]=="Finding the intersection, GLF")|
           (df["KC(Default)"]=="Plot pi")|
           (df["KC(Default)"]=="Plot whole number")|
           (df["KC(Default)"]=="Plot imperfect radical")|
           (df["KC(Default)"]=="Plot terminating proper fraction")|
           (df["KC(Default)"]=="Plot non-terminating improper fraction")|
           (df["KC(Default)"]=="Plot decimal - thousandths")|
           (df["KC(Default)"]=="Calculate part in proportion with fractions")|
           (df["KC(Default)"]=="Calculate unit rate")|
           (df["KC(Default)"]=="Calculate total in proportion with fractions")]

f = open("ct.csv", 'w+')
skill.to_csv(f)
