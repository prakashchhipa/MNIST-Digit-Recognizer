import math as mt
import numpy as np
import pandas as pd
import tensorflow as tflow
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


tflow.set_random_seed(0)

featuresTrain = None
featuresTrainCnnFrmt = None
lblTrainArr = None
featuresTestCnnFrmt = None
tfSession = None
trainStep = None
learningRate = None
kp = None
xEntropy = None
predictions = None
#train data 
def trainDataPrep():
    trainDF = pd.read_csv('data/train.csv')
    trainDFFeatures = trainDF.drop(['label'], axis=1)
    featuresTrain = trainDFFeatures.values.astype(dtype=np.float32)
    featuresTrainCnnFrmt = featuresTrain.reshape(42000, 28, 28, 1)
    trainLabels = trainDF['label'].tolist()
    ohTrainLabelsTensor = tflow.one_hot(trainLabels, depth=10)
    lblTrainArr = tflow.Session().run(ohTrainLabelsTensor).astype(dtype=np.float64)                    

#test data
def testDataPrep():
    testDF = pd.read_csv('data/test.csv')
    featuresTest = testDF.values.astype(dtype=np.float32)
    featuresTestCnnFrmt = featuresTest.reshape(28000, 28, 28, 1)

#batch
def retrieveBatch(iteration, size, trainFeatures, trainLabels):
    startIdx = (iteration * size) % 42000
    endIdx = startIdx + size
    batchX = trainFeatures[startIdx : endIdx]
    batchY = trainLabels[startIdx : endIdx]
    return batchX, batchY

def prepareCNNModel():
    Xrec = tflow.placeholder(tflow.float32, [None, 28, 28, 1])
    Y_lbl = tflow.placeholder(tflow.float32, [None, 10])
    learningRate = tflow.placeholder(tflow.float32)
    kp = tflow.placeholder(tflow.float32)

    #CNN- layer1 o/p
    L1 = 6
    #CNN- layer2 o/p
    L2 = 12
    #CNN- layer3 o/p
    L3 = 24
    #CNN- FC layer o/p
    L4_fc = 200

    W1 = tflow.Variable(tflow.truncated_normal([6, 6, 1, L1], stddev=0.1))  # 6x6 patch, 1 input channel, L1 output channels
    B1 = tflow.Variable(tflow.constant(0.1, tflow.float32, [L1]))
    W2 = tflow.Variable(tflow.truncated_normal([5, 5, L1, L2], stddev=0.1))
    B2 = tflow.Variable(tflow.constant(0.1, tflow.float32, [L2]))
    W3 = tflow.Variable(tflow.truncated_normal([4, 4, L2, L3], stddev=0.1))
    B3 = tflow.Variable(tflow.constant(0.1, tflow.float32, [L3]))

    W4 = tflow.Variable(tflow.truncated_normal([7 * 7 * L3, L4_fc], stddev=0.1))
    B4 = tflow.Variable(tflow.constant(0.1, tflow.float32, [L4_fc]))
    W5 = tflow.Variable(tflow.truncated_normal([L4_fc, 10], stddev=0.1))
    B5 = tflow.Variable(tflow.constant(0.1, tflow.float32, [10]))

    #28x28
    stride = 1 
    Y1 = tflow.nn.relu(tflow.nn.conv2d(Xrec, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    #14x14
    stride = 2
    Y2 = tflow.nn.relu(tflow.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    #7x7
    stride = 2
    Y3 = tflow.nn.relu(tflow.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

    #L3 to L4_fc transfromation
    YY = tflow.reshape(Y3, shape=[-1, 7 * 7 * L3])

    Y4 = tflow.nn.relu(tflow.matmul(YY, W4) + B4)
    YY4 = tflow.nn.dropout(Y4, kp)
    Y_logits = tflow.matmul(YY4, W5) + B5
    Y = tflow.nn.softmax(Y_logits)

    xEntropy = tflow.nn.softmax_cross_entropy_with_logits(logits=Y_logits, labels=Y_lbl)
    xEntropy = tflow.reduce_mean(xEntropy)*100
    rightPredict = tflow.equal(tflow.argmax(Y, 1), tflow.argmax(Y_lbl, 1))
    acc = tflow.reduce_mean(tflow.cast(rightPredict, tflow.float32))
    predictions = tflow.argmax(Y, 1)
    #optimizer- Adam
    trainStep = tflow.train.AdamOptimizer(learningRate).minimize(xEntropy)

def initializeCNNModel():
    init = tflow.global_variables_initializer()
    tfSession = tflow.Session()
    tfSession.run(init)


def trainM(iteration):

    #100 image at once, it could be changed
    size = 100
    batchX, batchY = retrieveBatch(iteration, size, featuresTrainCnnFrmt, lblTrainArr)
    maxLR = 0.0032
    minLR = 0.0001
    decaySpeed = 2000.00
    LR = minLR + (maxLR - minLR) * mt.exp(-iteration/decaySpeed)

    #test
    if iteration % 100 == 0:
        accr, loss = tfSession.run([acc, xEntropy], {Xrec: featuresTrainCnnFrmt[-10000:], Y_lbl: lblTrainArr[-10000:], kp: 1.0})
        print(str(iteration) + ": test accuracy:" + str(accr) + " test loss: " + str(loss))
    #train
    if iteration % 20 == 0:
        accr, loss = tfSession.run([acc, xEntropy], {Xrec: batchX, Y_lbl: batchY, kp: 1.0})
        print(str(iteration) + ": train accuracy:" + str(accr) + " train loss: " + str(loss))

    tfSession.run(trainStep, {Xrec: batchX, Y_lbl: batchY, learningRate: LR, kp: 0.75})


def modelTraning():
    trainDataPrep()
    testDataPrep()
    prepareCNNModel()
    initializeCNNModel()
    for iteration in range(100001): 
        trainM(iteration)

def modelTesting():
    a, c = tfSession.run([acc, xEntropy], {Xrec: featuresTrainCnnFrmt[-10000:], Y_lbl: lblTrainArr[-10000:], kp: 1.0})
    print("\nAccuracyTest:" + str(a) + "Loss: " + str(c))

def predictTest():
    # Get predictions on test data
    p = tfSession.run([predictions], {Xrec: featuresTestCnnFrmt, kp: 1.0})

    # Write predictions to csv file
    testResult = pd.DataFrame({'ImageId': pd.Series(range(1, len(p[0]) + 1)), 'Label': pd.Series(p[0])})
    testResult.to_csv('results.csv', index=False)


#main method

if __name__="__main__":
    modelTraning()
    modelTesting()
    predictTest()








