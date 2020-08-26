#script to find the 10 skills with the most data in assistments for the SKILL BUILDER DATA SET, referencing the kt-idem paper
import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
import io
import requests

def find_skills():
    url = "data/ct_bta.csv"
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
    skill_list = df['KC(SubSkills)'].value_counts()[150:-300].index.tolist() #skills between 700-2000 data points
    #test_skills = np.random.choice(skill_list, 12)
    test_skills = ["[SkillRule: Isolate positive; x+a=b, positive]",
    "simplify-fractions-sp",
    "Identify no more factors",
    "Calculate percent out of context",
    "Enter improper fraction from given model",
    "Enter numerator of percent change with variable",
    "Compare fractions from contextual problem",
    "combine-like-terms-sp",
    "Identify fraction using number line",
    "Calculate sum with negative integer",
    "[SkillRule: Isolate negative; x+a=b, negative]",
    "Calculate percent from given decimal"]

    student_count = []
    data_count = []
    template_count = []
    for i in test_skills:
        df_temp = df[(df["KC(SubSkills)"] == i)]
        student_count.append(df_temp["Anon Student Id"].nunique())
        data_count.append(len(df_temp))
        template_count.append(df_temp["Problem Name"].nunique())
    return df, test_skills, student_count, data_count, template_count
