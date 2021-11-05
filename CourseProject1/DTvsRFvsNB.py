# David Vadnais
import numpy as np
import csv
import pandas
import timeit
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
import sys

############################################################
def readTrain(training_file):
    df = pandas.read_csv(training_file, header=None)
    df.fillna('null', inplace=True)
    print(df) 

    x=df.iloc[: , 0:-2]
    y=df.iloc[: , -1]

    #make data numeric
    xn = x.to_numpy()
    print(x)
    newArray = []

    for col in range(len(x.columns)):
        tmpCol = []
        #n = col
        for value in range(len(xn)):
            s = str(xn[value][col]).lower().strip()

            tmpCol.append(hash(s))

        newArray.append(tmpCol)

    X = pandas.DataFrame({'0': newArray[0]})
    features = ['0']
    for i in range(len(x.columns)):
        if (i==0):
            continue
        X.insert(loc=i, column=str(i), value=newArray[i])
        features.append(str(i))

    #DO Y
    yn = y.to_numpy()

    d = dict([(ytmp,xtmp+1) for xtmp,ytmp in enumerate(sorted(set(yn[:])))])
    if not (yn.any() == 'null'):# add a null case in case it comes up later
        d['null'] = len(d)
        
    tmpCol = [d[xtmp] for xtmp in yn[:]]
    Y = pandas.DataFrame({'0': tmpCol})
    
    return X, Y, d
############################################################
def readTest(training_file, dictionaryFromTest):
    df = pandas.read_csv(test_file, header=None)
    df.fillna('null', inplace=True)
    
    x=df.iloc[: , 0:-2]
    y=df.iloc[: , -1]
    #make data numeric
    xn = x.to_numpy()

    newArray = []

    for col in range(len(x.columns)):
        tmpCol = []
        #n = col
        for value in range(len(xn)):
            s = str(xn[value][col])
            tmpCol.append(str(hash(s.lower())))

        newArray.append(tmpCol)

    X_test = pandas.DataFrame({'0': newArray[0]})
    features = ['0']
    for i in range(len(x.columns)):
        if (i==0):
            continue
        X_test.insert(loc=i, column=str(i), value=newArray[i])
        features.append(str(i))

    #DO Y
    yn = y.to_numpy()

    tmpCol = [dictionaryFromTest[xtmp] for xtmp in yn[:]]
    Y_test = pandas.DataFrame({'0': tmpCol})
    return X_test, Y_test
############################################################
def printResults(y_pred,Y_test,d):
    val_list = list(d.values())
    key_list = list(d.keys())
    Y_test_np = Y_test.to_numpy()
    for i in range(len(y_pred)):
        thisid = i+1
        thispred = key_list[val_list.index(y_pred[i])]
        thisTruth = key_list[val_list.index(Y_test_np[i])]
        acc = 1
        if(y_pred[i] != Y_test_np[i]):
            acc = 0
        print('ID='+str(thisid)+", predicted="+thispred+", true="+thisTruth \
              +", accuracy="+str(acc))
    print("\n")
    confMat = confusion_matrix(Y_test, y_pred)
    print(confMat)
    print("\n")
    numberRight = confMat[0,0]+confMat[1,1]
    accAll = numberRight/len(y_pred)
    print("accuracy = "+str(accAll)+"\n")
    
    TP = confMat[0,0]
    FP = confMat[0,1]
    prec = TP/(TP+FP)
    print("precision = "+str(prec)+"\n")
    FN = confMat[1,0]
    recall = TP/(TP+FP) 
    print("recall = "+str(recall)+"\n")
    return accAll, prec, recall
############################################################
def decision_tree(training_file,test_file):
    
    X, Y, d = readTrain(training_file)
    
    start = timeit.default_timer()
    
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, Y)
    
    X_test, Y_test = readTest(training_file, d)
    y_pred = dtree.predict(X_test)
    
    stop = timeit.default_timer()
    
    return printResults(y_pred,Y_test,d), stop - start
    
############################################################
def random_forest(training_file,test_file):
    X, Y, d = readTrain(training_file)

    start = timeit.default_timer()
    
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X,Y.to_numpy()[:,0])
    X_test, Y_test = readTest(training_file, d)
    y_pred = clf.predict(X_test)
    
    stop = timeit.default_timer()

    return printResults(y_pred,Y_test,d), stop - start
    
############################################################
def naive_bayes(training_file,test_file):
    X, Y, d = readTrain(training_file)
    
    start = timeit.default_timer()
    nb = GaussianNB()
    nb.fit(X,Y.to_numpy()[:,0])
    X_test, Y_test = readTest(training_file, d)
    y_pred = nb.predict(X_test)
    
    stop = timeit.default_timer()

    return printResults(y_pred,Y_test,d), stop - start

############################################################
#training_file = 'dataSplit/credit_trainset.txt'
#test_file = 'dataSplit/credit_testset.txt'

#training_file = 'dataSplit/census_trainset.txt'
#test_file = 'dataSplit/census_testset.txt'

#training_file ='dataSplit/Task6_credit_trainset.txt'
#test_file = 'dataSplit/Task6_credit_testset.txt'
print("You are running : "+ sys.argv[1]+ "\nAuthor: David Vadnais\n")

if len(sys.argv) != 3:
    raise ValueError('Please provide 2 data files')
    


training_file = sys.argv[1]
test_file = sys.argv[2]
    
r1, et1 =decision_tree(training_file,test_file)

r2, et2 = random_forest(training_file,test_file)

r3, et3 = naive_bayes(training_file,test_file)

print("    , dt                 , rf                 , nb \n")
print("accu, " + str(r1[0]) + " , "+str(r2[0]) + " , "+str(r3[0]) )
print("prec, " + str(r1[1]) + " , "+str(r2[1]) + " , "+str(r3[1]) )
print("reca, " + str(r1[2]) + " , "+str(r2[2]) + " , "+str(r1[2]) )
print("time, " + str(et1)+ " , "+str(et2) + " , "+str(et3))
