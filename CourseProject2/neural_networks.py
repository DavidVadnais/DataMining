# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Process data
(x_tmp, y), (x_test, y_test) = cifar10.load_data()


x_tmp = x_tmp/255 # normalize


x = np.zeros((x_tmp.shape[0] , x_tmp.shape[1]*x_tmp.shape[2]*x_tmp.shape[3]))
# flatten x
for image in range(x_tmp.shape[0]):
    for row in range(x_tmp.shape[1]):
        for colum in range(x_tmp.shape[2]):
            for color in range(x_tmp.shape[3]):
                x[image]=x_tmp[image , row, colum, color]
    if image % 10000 == 0:
        print('% done :' ,image/x_tmp.shape[0])
                
print('x shape:', x_tmp.shape)
print(x.shape[0], 'train samples')
print(x.shape[0], 'test samples')

print('x')
print(x)

print('y')
print(y)

def ReLu(x):
    #return np.where(x >= 0, x, 0)
    return np.maximum(x, 0)/255
  
def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

def ReLuDer(x):
  #ret = []
  #for i in range(len(x)):
  #    if x[i] < 0:
  #        ret.append(1)
  #    else :
  #        ret.append(0)
  ret = np.where(x >= 0, 1, 0)
  return ret

def softmax(A):
  expA = np.exp(A)
  return expA / np.sum(expA, axis = 1, keepdims = True)

  #return expA / expA.sum(axis=1, keepdims=True)
  
 def lossSumOfSquare(y,y_hat):
    #loss = 0
    #for i in range(len(y)):
    loss = 1/2*(y-y_hat)**2
    
    return loss
def derLoss(y,y_hat):
    derloss = -(y-y_hat)
    return derloss
  
def loadData_Tokenizer(X_train, X_test,MAX_NB_WORDS=75000,MAX_SEQUENCE_LENGTH=500):
    np.random.seed(7)
    text = np.concatenate((X_train, X_test), axis=0)
    text = np.array(text)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Found %s unique tokens.' % len(word_index))
    indices = np.arange(text.shape[0])
    # np.random.shuffle(indices)
    text = text[indices]
    print(text.shape)
    X_train = text[0:len(X_train), ]
    X_test = text[len(X_train):, ]
    embeddings_index = {}
    f = open(".\\Glove\\glove.6B.50d.txt", encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            pass
        embeddings_index[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))
    return (X_train, X_test, word_index,embeddings_index)
#############################################
# Driver
#############################################
np.random.seed(100)
output_labels = 10#Returns # of unique elements in y
hNodes = 4
numbEpochs = 1000
step_size =  0.0000005
    
howManyExamples = x.shape[0]
outputs = x.shape[1]

#Transform
yT = np.zeros((len(y) , output_labels))
for i in range(len(y)):
    yT[i,y[i]]=1

numInNodes = x.shape[1]

###########begin
errorForPlot = []
accForPlot = []

wh = np.random.rand(numInNodes,hNodes)  # Weights
wo = np.random.rand(hNodes,output_labels)

bh = np.random.randn(hNodes)    # Bias
bo = np.random.randn(output_labels)

for epoch in range(numbEpochs):
    ##################### forward #######################

    # forward 1
    tempH = np.dot(x, wh) + bh
    nodeValueH = ReLu(tempH)
    
    # forward 2
    tempO = np.dot(nodeValueH, wo) + bo
    nodeValO = softmax(tempO)

    ##################### backward #######################
    # back 1

    dCostdTempO = nodeValO - yT

    dCostdwo = np.dot(nodeValueH.T, dCostdTempO)

    # back 2
    dCostdnodeValueH = np.dot(dCostdTempO , wo.T)
    dnodeValueH_dtempH = ReLuDer(tempH)
    dCostdwh = np.dot(x.T, dnodeValueH_dtempH * dCostdnodeValueH)

    dCostdbh = dCostdnodeValueH * dnodeValueH_dtempH
    
    # Fix weights for next pass
    wh -= step_size * dCostdwh
    wo -= step_size * dCostdwo
    
    #fix bias for next pass
    bh -= step_size * dCostdbh.sum(axis=0)
    bo -= step_size * dCostdTempO.sum(axis=0)

    if epoch % (numbEpochs/100) == 0:
        loss = np.sum(-yT * np.log(nodeValO))
        errorForPlot.append(loss)
        
        tmp = 0
        for i in range(len(y)):
            a = np.argmax(nodeValO[i,:])
            if a == y[i]:
                tmp += 1
        
        accuracy = tmp/len(y)
        accForPlot.append(accuracy)

        print('Loss : ', loss)
        print('acc : ', accuracy)

        
        
#############################################        
# Plotting
#############################################

fig = plt.figure()
plt.title('Our ReLu ANN Accuracy vs Epoch for CIFAR-10')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()

xPlot = range(len(accForPlot))
plt.scatter(xPlot , accForPlot)
plt.show()

# Loss

fig2 = plt.figure
plt.title('Our ReLu ANN Loss vs Epoch for CIFAR-10')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()

xPlot = range(len(errorForPlot))
plt.scatter(xPlot , errorForPlot)
plt.show()
