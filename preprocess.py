'''
We need to classify every sentance in a job description as:
1. required skills
2. required years of experience
3. required degree
4. company culture
5. desired by not required

This script preprocesses a monster jobs csv file into raw text lines which are manually labeled later
'''
import pandas as pd
import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\*", "", string)
    return string.strip().lower()

def containsList(lSmall, lBig):
    for i in xrange(len(lBig)-len(lSmall)+1):
        for j in xrange(len(lSmall)):
            if lBig[i+j] != lSmall[j]:
                break
        else:
            return i, i+len(lSmall)
    return False, False


df = pd.read_csv('./data/monster-jobs/monster_com-job_sample.csv')
# sDelimiters = "\xe2\x80\xa2", ". ", "! ", "\n", "..." # probably not a complete list
# sRegexPattern = '|'.join(map(re.escape, sDelimiters))

# TODO: send this to that online platform - 100 thru 200 are staged
iCount = 0
iSearchRange = 15
lCategories = ['benefits','culture','degree','desired','other','required','seeking','tasks','years']
lKeywords = [['401k', '401 k', 'stock option', 'benefits', 'dental'],['equal opportunity'],['degree', 'degrees', "Associate of Arts","A.A.","AA","Associate of Science","A.S.","AS","Associate of Applied Science","AAS","Bachelor of Arts","B.A.","BA","Bachelor of Science","B.S.","BS","Bachelor of Fine Arts","BFA","Bachelor of Applied Science","BAS","General Education Development","GED","High school diploma","Master of Arts","M.A.","MA","Master of Science","M.S.","MS","Master of Business Administration","MBA","Master of Fine Arts","MFA","Doctor of Philosophy","Ph.D.","PhD","Juris Doctor","J.D.","JD","Doctor of Medicine","M.D.","MD","Doctor of Dental Surgery","DDS"],['is a plus', 'is preferred', 'plus', 'preferred'],[],['requires', 'required', 'must'],['seeking', 'is looking for'],['duty'],['years', 'year']]

for sCategorie in lCategories:
    oTextFile = open("./data/monster-jobs/" + sCategorie + "_new.txt", "w")
    for sDescription in df.job_description:
        lWords = sDescription.split() # split degree to array
        for sKeyword in lKeywords[iCount]:
            if sKeyword in lWords:
                iFoundIndex = lWords.index(sKeyword)
                lNearbyWords = lWords[iFoundIndex-iSearchRange:iFoundIndex] + lWords[iFoundIndex:iFoundIndex+iSearchRange] #
                sNearbyWords = " ".join(lNearbyWords)
                sNearbyWords = clean_str(sNearbyWords) # for the machine learning algo
                oTextFile.write(sNearbyWords + '\n')
    oTextFile.close()
    iCount = iCount + 1
