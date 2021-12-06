from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import timeit

from keras.preprocessing.text import Tokenizer
import glob
import pandas as pd
import numpy as np
from math import sqrt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from collections import defaultdict

import sys

training_file = sys.argv[1]
test_file = sys.argv[2]
wordLimit = 5000


root_dir=training_file
root_dir='./20_newsgroups_Train'
root_dirL = len(root_dir)+1
# root_dir needs a trailing slash (i.e. /root/dir/)
i = 0
train = []#pd.DataFrame(columns=['folder', 'fileName'])
for filename in glob.iglob(root_dir + '**/**/*', recursive=True):
    if i > 19:
        for j, c in enumerate(filename[root_dirL:]):
            if c.isdigit():
                bs = j
                #print(bs)
                break
        #print(filename[22:])
        #print(filename[22:22+bs-1])#lable
        label = filename[root_dirL:root_dirL+bs-1]
        #print(filename[22+bs:])#filename
        filenametmp = filename[root_dirL+bs:]
        train.append([label,filenametmp])
    #print(filename)
    i+=1
    
root_dirTest=test_file
root_dirTest='./20_newsgroups_Test'
root_dirTL = len(root_dirTest)+1
# root_dir needs a trailing slash (i.e. /root/dir/)
i = 0
test = []#pd.DataFrame(columns=['folder', 'fileName'])
for filename in glob.iglob(root_dirTest + '**/**/*', recursive=True):
    if i > 19:
        for j, c in enumerate(filename[root_dirTL:]):
            if c.isdigit():
                bs = j
                #print(bs)
                break
        #print(filename[22:])
        #print(filename[22:22+bs-1])#lable
        label = filename[root_dirTL:root_dirTL+bs-1]
        #print(filename[22+bs:])#filename
        filenametmp = filename[root_dirTL+bs:]
        test.append([label,filenametmp])
    #print(filename)
    i+=1
    
allDocsAsStringsTrain=[]
allDocsAsStringsNoLabelsTrain = []
lastrow=""

category = -1
ytest = []
ytrain = []
for row in train:
    fileToOpen= root_dir+'/'+row[0]+'/'+row[1]
    
    file=open(fileToOpen,"r")
    tmp = file.read().lower()#text_cleaner(file.read()).lower()
    
    if (row[0]!= lastrow):
        lastrow = row[0]
        category += 1
    allDocsAsStringsTrain.append([row[0],row[1],tmp])
    ytrain.append(category)
    allDocsAsStringsNoLabelsTrain.append(tmp)
    file.close()
    
allDocsAsStringsTest=[]
allDocsAsStringsNoLabelsTest = []
lastrow=""
category = -1
for row in test:
    fileToOpen= root_dirTest+'/'+row[0]+'/'+row[1]
    if (row[0]!= lastrow):
        lastrow = row[0]
        category+=1
    
    #print(fileToOpen)
    file=open(fileToOpen,"r")
    tmp = file.read().lower()#text_cleaner(file.read()).lower()
    ytest.append(category)
    allDocsAsStringsTest.append([row[0],row[1],tmp])
    allDocsAsStringsNoLabelsTest.append(tmp)
    file.close()
    
corpus = []
i=0
#ytrain = []
for doc in allDocsAsStringsNoLabelsTrain:
    corpus.append(''.join((element for element in doc if not (element.isdigit() or element =='_'))))
    #if i%10 == 0:
    #    print(doc)
    #ytrain.append(allDocsAsStringsTrain[i][3])
    i=+1
    
#ytest = []  
corpusTest = []
i=0
for doc in allDocsAsStringsNoLabelsTest:
    corpusTest.append(''.join((element for element in doc if not (element.isdigit() or element =='_'))))
    #ytest.append(allDocsAsStringsTest[i][3])

    i=+1
ytrain=np.array(ytrain)
ytest =np.array(ytest)

bows= []
bowsCol = []
cat =0
i = 0
while i < len(corpus)-1:
    tmp = []
    while cat == ytest[i]:
        if (ytest[i]%10 == 0): print(ytest[i])
        #print('cat' + str(cat))
        tmp.append(corpus[i])
        i+=1
        if i == len(corpus)-1:
            break
        #print(i)
    #print('yeet')
    cv = CountVectorizer(stop_words='english', lowercase=True,max_features=wordLimit)
    word_count = cv.fit_transform(tmp) # Fit the model
    bowsCol.append(cv.get_feature_names_out())
    bows.append(pd.DataFrame(word_count.toarray(), columns = cv.get_feature_names_out()))
    cat += 1
    print(cat)
    
#prob cat = 
sums = []
allws = 0
for bow in bows:
    sums.append(bow.sum(axis=1))
    allws += bow.sum(axis=1)

pCat = []
for asum in sums:
    pCat.append(asum/allws)
start = timeit.default_timer()

i=0
preds = []
while i < len(corpusTest)-1:
    doc = corpusTest[i]
    word_count = cv.fit_transform([doc]) # Fit the model
    words = pd.DataFrame(word_count.toarray(), columns = cv.get_feature_names_out())
    wordsNames = cv.get_feature_names_out()
    
    j= 0
    pfileIsInCat = 0
    while j < len(bows):
        thisBow = bows[j]
        word = 0 
        #print(words.shape[1])
        wordsnp = np.array(words)
        pWord = 0
        for wordI in range(words.shape[1]):
            if wordsNames[wordI] not in bow.columns:
                continue
            
            #colOfWord = thisBow.columns.get_loc(wordsNames[wordI])
            pab = bow[wordsNames[wordI]].sum()
            #words.
            pWord += pCat[j]*pab
        pfileIsInCat += pWord
        j+=1
    preds.append(np.argmax(pfileIsInCat))
    if(i%10 == 0):
        print(i)
    i+=1
stop = timeit.default_timer()
#elapsed = stop-start
#print(elapsed)

cases = len(preds)

index = 0
right = 0
wrong = 0
recalls = np.zeros([category+1,2])

for i in preds:
    j=-1
    
    for apred in i:
        if Apred == int(ytest[index]):
            recalls[pred][0]+=1
            right +=1
        else:
            recalls[pred][1]+=1
            wrong+=1
        index+=1
    
print(wrong)
print(right/(right+wrong))

rec = 0
for row in recalls:
    rec+=(row[0]/(row[0]+row[1])) 
rec = rec/(category+1)

print('wrong: '+str(wrong))
print('precision: '+str(right/(right+wrong)))
print('elapsed time: '+str(elapsed))
print('recall: '+str(rec))