import numpy as np
import csv
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB



############################################################
def readTrain(training_file):
    df = pandas.read_csv(training_file, header=None)
    print(df) 

    x=df.iloc[: , 0:-2]
    y=df.iloc[: , -1]

    #make data numeric
    xn = x.to_numpy()

    newArray = []

    for col in range(len(x.columns)):
        tmpCol = []
        #n = col
        for value in range(len(xn)):
            s = str(xn[value][col]).lower()
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
    tmpCol = [d[xtmp] for xtmp in yn[:]]
    Y = pandas.DataFrame({'0': tmpCol})
    
    return X, Y, d
############################################################
def readTest(training_file, dictionaryFromTest):
    df = pandas.read_csv(test_file, header=None)

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
            tmpCol.append(hash(s.lower()))

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
    print(confusion_matrix(Y_test, y_pred))
############################################################
def decision_tree(training_file,test_file):
    
    X, Y, d = readTrain(training_file)
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, Y)
    

    X_test, Y_test = readTest(training_file, d)
    y_pred = dtree.predict(X_test)
    
    printResults(y_pred,Y_test,d)
    
############################################################
def random_forest(training_file,test_file):
    X, Y, d = readTrain(training_file)

    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X,Y.to_numpy()[:,0])
    X_test, Y_test = readTest(training_file, d)
    y_pred = clf.predict(X_test)
    

    printResults(y_pred,Y_test,d)
    
    
def naive_bayes(training_file,test_file):
    X, Y, d = readTrain(training_file)

    nb=RandomForestClassifier(n_estimators=100)
    nb.fit(X,Y.to_numpy()[:,0])
    X_test, Y_test = readTest(training_file, d)
    y_pred = nb.predict(X_test)
    

    printResults(y_pred,Y_test,d)

############################################################
training_file = 'dataSplit/credit_trainset.txt'
test_file = 'dataSplit/credit_testset.txt'
training_file = 'dataSplit/census_trainset.txt'
test_file = 'dataSplit/census_testset.txt'
decision_tree(training_file,test_file)

random_forest(training_file,test_file)

naive_bayes(training_file,test_file)