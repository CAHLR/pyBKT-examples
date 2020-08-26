#script to find the 10 skills with the most data in assistments for the SKILL BUILDER DATA SET, referencing the kt-idem paper
import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
import io
import requests

def find_skills():
    url = "data/as.csv"
    # save string only after last slash for file name
    urltofile = url.rsplit('/', 1)[-1]

    # if url is a local file, read it from there
    if os.path.exists("data/"+urltofile):
        try:
            f = open("data/" + urltofile, "rb")
            # assume comma delimiter
            df = pd.read_csv(io.StringIO(f.read().decode('latin')), low_memory=False)
        except:
            f = open("data/" + urltofile, "rb")
            # try tab delimiter if comma delimiter fails
            df = pd.read_csv(io.StringIO(f.read().decode('latin')), low_memory=False, delimiter='\t')
            
    df = df[(df["original"]==1)]

    #skill_list = df['skill_name'].value_counts()[:10].index.tolist()
    skill_list = [
    "Percent Of",
    "Addition and Subtraction Integers",
    "Conversion of Fraction Decimals Percents",
    "Volume Rectangular Prism",
    "Venn Diagram",
    "Equation Solving Two or Fewer Steps",
    "Volume Cylinder",
    "Multiplication and Division Integers",
    "Area Rectangle",
    "Addition and Subtraction Fractions",
    ]
    student_count = []
    data_count = []
    template_count = []
    for i in skill_list:
        df_temp = df[(df["skill_name"] == i)]
        student_count.append(df_temp["user_id"].nunique())
        data_count.append(len(df_temp))
        template_count.append(df_temp["template_id"].nunique())
    return df, skill_list, student_count, data_count, template_count
#for i in skill_list:
#    df_temp = df[(df["skill_name"] == i)]
#    print("# students for skill", i, ":", df_temp["user_id"].nunique())#
 #   print("# templates for skill", i, ":", df_temp["template_id"].nunique())

#return skill_list
