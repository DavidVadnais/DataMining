from sklearn.datasets import fetch_20newsgroups
from keras.layers import  Dropout, Dense
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
import timeit

import sys


from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
import glob
import pandas as pd
import numpy as np
from math import sqrt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from collections import defaultdict

training_file = sys.argv[1]
test_file = sys.argv[2]

def TFIDF(X_train, X_test,MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print("tf-idf with",str(np.array(X_train).shape[1]),"features")
    return (X_train,X_test)

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

X_train_tfidf,X_test_tfidf = TFIDF(corpus,corpusTest)

model = Sequential()
node = 512 # number of nodes
nLayers = 3 # number of  hidden layer
dropout=0.5
shape = X_train_tfidf.shape[1]
nClasses = 20
model.add(Dense(node,input_dim=shape,activation='relu'))
model.add(Dropout(dropout))
for i in range(0,nLayers):
    model.add(Dense(node,input_dim=node,activation='relu'))
    model.add(Dropout(dropout))
model.add(Dense(nClasses, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(X_train_tfidf, ytrain,
                              validation_data=(X_test_tfidf, ytest),
                              epochs=10,
                              batch_size=128,
                              verbose=2)

start = timeit.default_timer()
predicted = model.predict(X_test_tfidf)
stop = timeit.default_timer()

elapsed = stop-start

cases = predicted.shape[1]

index = 0
right = 0
wrong = 0
recalls = np.zeros([category+1,2])
for i in predicted:
    j=-1
    
    pred = np.argmax(i)
    
    if pred == int(ytest[index]):
        recalls[pred][0]+=1
        right +=1
    else:
        recalls[pred][1]+=1
        wrong+=1
    index+=1
    
rec = 0
for row in recalls:
    rec+=(row[0]/(row[0]+row[1])) 
rec = rec/(category+1)

print('wrong: '+str(wrong))
print('precision: '+str(right/(right+wrong)))
print('elapsed time: '+str(elapsed))
print('recall: '+str(rec))