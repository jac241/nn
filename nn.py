
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, SoftmaxLayer
from pybrain.tools.validation import CrossValidator, ModuleValidator
import numpy as np
import random
from pybrain.supervised.trainers import BackpropTrainer

n = FeedForwardNetwork()
#inLayer = LinearLayer(27)
outLayer = SoftmaxLayer(2)
#hiddenLayer = SigmoidLayer(30)
#n.addInputModule(inLayer)
#n.addModule(hiddenLayer)
n.addOutputModule(outLayer)

#dyspnea, cough, haemoptysis, PAD, MI, Asthma
dypsneaLayer = LinearLayer(1)
coughLayer = LinearLayer(1)
haemoLayer = LinearLayer(1)
padLayer = LinearLayer(1)
miLayer = LinearLayer(1)
asthmaLayer = LinearLayer(1)

#Type 2 Diabetes, Smoking
#allLayer = LinearLayer(2)
diabetesLayer = LinearLayer(1)
smokingLayer = LinearLayer(1)

#diagnosis
diagnosisAllLayer = LinearLayer(7)
#performance goes to other layer
performanceStatusLayer = LinearLayer(3)

#Tumor
tAllLayer = LinearLayer(4)

#FEV, Weakness, Pain, Age, FVC
fevLayer = LinearLayer(1)
weaknessLayer = LinearLayer(1)
painLayer = LinearLayer(1)
ageLayer = LinearLayer(1)
fvcLayer = LinearLayer(1)

###########Hidden Layers......
coughGroupLayer = SigmoidLayer(2)
diabetesGroupLayer = SigmoidLayer(1)
perfomanceHiddenLayer = SoftmaxLayer(1)
tumorDiagHiddenLayer = SigmoidLayer(4)

withPerformanceLayer = SigmoidLayer(6)

bigFinalHiddenLayer = SigmoidLayer(20)
##########################

n.addInputModule(diagnosisAllLayer)
n.addInputModule(fvcLayer)
n.addInputModule(fevLayer)
n.addInputModule(performanceStatusLayer)
n.addInputModule(painLayer)
n.addInputModule(haemoLayer)
n.addInputModule(dypsneaLayer)
n.addInputModule(coughLayer)
n.addInputModule(weaknessLayer)
n.addInputModule(tAllLayer)
n.addInputModule(diabetesLayer)
n.addInputModule(miLayer)
n.addInputModule(padLayer)
n.addInputModule(smokingLayer)
n.addInputModule(asthmaLayer)
n.addInputModule(ageLayer)

n.addModule(coughGroupLayer)
n.addModule(diabetesGroupLayer)
n.addModule(perfomanceHiddenLayer)
n.addModule(tumorDiagHiddenLayer)
n.addModule(withPerformanceLayer)
n.addModule(bigFinalHiddenLayer)


from pybrain.structure import FullConnection
#n.addConnection(FullConnection(inLayer, hiddenLayer))
#n.addConnection(FullConnection(hiddenLayer, outLayer))
n.addConnection(FullConnection(coughLayer, coughGroupLayer))
n.addConnection(FullConnection(dypsneaLayer, coughGroupLayer))
n.addConnection(FullConnection(haemoLayer, coughGroupLayer))
n.addConnection(FullConnection(padLayer, coughGroupLayer))
n.addConnection(FullConnection(miLayer, coughGroupLayer))
n.addConnection(FullConnection(asthmaLayer, coughGroupLayer))

n.addConnection(FullConnection(diabetesLayer, diabetesGroupLayer))
n.addConnection(FullConnection(smokingLayer, diabetesGroupLayer))

n.addConnection(FullConnection(performanceStatusLayer, perfomanceHiddenLayer))

n.addConnection(FullConnection(diagnosisAllLayer, diabetesGroupLayer))
n.addConnection(FullConnection(tAllLayer, diabetesGroupLayer))

n.addConnection(FullConnection(coughGroupLayer, withPerformanceLayer))
n.addConnection(FullConnection(diabetesGroupLayer, withPerformanceLayer))
n.addConnection(FullConnection(perfomanceHiddenLayer, withPerformanceLayer))

n.addConnection(FullConnection(withPerformanceLayer, bigFinalHiddenLayer))
n.addConnection(FullConnection(tumorDiagHiddenLayer, bigFinalHiddenLayer))

n.addConnection(FullConnection(bigFinalHiddenLayer, outLayer))

n.sortModules()

def getMaxIndex(myList):
    index = 0
    for i in range(len(myList)):
        if myList[i] > myList[index]:
            index = i
    return index

#print "The dimension of output by net:" + str(n.outdim)

from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
ds = ClassificationDataSet(27, 2)

file = open('BinarySubset.txt', 'r')
ourData = []

features = []
classes = []

for line in file:
    inputData = line.split(',') #put training example data into list
    classID = inputData.pop() #last element is the classification, pop it off the data list
    for i in range(len(inputData)):
        inputData[i] = float(inputData[i])
    classVec = [0] * 2
    if classID == 'T\n' or classID == 'T':
        classVec[0] = 1
    else:
        classVec[1] = 1
    ds.addSample(inputData, classVec)
    features.append(inputData)
    classes.append(classVec)


# test_ds, trndata = ds.splitWithProportion(0.25)
# modval = ModuleValidator()
# #print "Number of training patterns: ", len(trndata)
# #print "Input and output dimensions: ", trndata.indim, trndata.outdim
# #print "First sample (input, target, class):"
# #print trndata['input'][0], trndata['target'][0] #, trndata['class'][0]
# 
# numCorrect = 0
# print("Before training")
# for data in test_ds:
#     input_entry = data[0]
#     output_entry = data[1]
#     pred = n.activate(input_entry)
#     print 'Actual:', output_entry, 'Predicted', pred
#     if getMaxIndex(pred) == getMaxIndex(output_entry):
#         numCorrect += 1
# 
# correctRate = float(numCorrect) / len(test_ds)
# print "Neural network success rate: " + str(correctRate)
# 
# 
# trainer = BackpropTrainer(n, trndata)
# trainer.trainEpochs(1000)
# 
# numCorrect = 0
# print("After training")
# for data in test_ds:
#     input_entry = data[0]
#     output_entry = data[1]
#     pred = n.activate(input_entry)
#     print 'Actual:', output_entry, 'Predicted', pred 
#     if getMaxIndex(pred) == getMaxIndex(output_entry):
#         numCorrect += 1
# 
# 
# correctRate = float(numCorrect) / len(test_ds)
# 
# print "Neural network success rate: " + str(correctRate)
# 
# 
# 
# cValidator = CrossValidator(trainer, trndata, n_folds=5, valfunc=modval.MSE)
# print "MSE %f: " %(cValidator.validate())


n_folds = 10    
perms = np.array_split(np.arange(len(features)), n_folds)
results = []
for i in xrange(n_folds):
    
    n.reset()
    
    train_ds = ClassificationDataSet(27, 2)
    test_ds = ClassificationDataSet(27, 2)
    train_perms_idxs = range(n_folds)
    train_perms_idxs.pop(i)
    temp_list = []
    for train_perms_idx in train_perms_idxs:
        temp_list.append(perms[ train_perms_idx ])
    train_idxs = np.concatenate(temp_list)
    
    for idx in train_idxs:
        train_ds.addSample(features[idx], classes[idx])

    # determine test indices
    test_idxs = perms[i]
    for idx in test_idxs:
        test_ds.addSample(features[idx], classes[idx])
    
    numCorrect = 0
    print("Before training Fold %d" % i)
    for data in test_ds:
        input_entry = data[0]
        output_entry = data[1]
        pred = n.activate(input_entry)
        print 'Actual:', output_entry, 'Predicted', pred
        if getMaxIndex(pred) == getMaxIndex(output_entry):
            numCorrect += 1

    correctRate = float(numCorrect) / len(test_ds)
    print "Neural network success rate: " + str(correctRate)
    
    
    trainer = BackpropTrainer(n, train_ds)
#     trainer.trainEpochs(500)
    trainer.trainEpochs(2000)
    
    numCorrect = 0
    print("After training Fold %d" % i)
    for data in test_ds:
        input_entry = data[0]
        output_entry = data[1]
        pred = n.activate(input_entry)
        print 'Actual:', output_entry, 'Predicted', pred 
        if getMaxIndex(pred) == getMaxIndex(output_entry):
            numCorrect += 1
    
    
    correctRate = float(numCorrect) / len(test_ds)
    
    print "Neural network success rate: " + str(correctRate)
    
    print "\n\n\n\n"
    

