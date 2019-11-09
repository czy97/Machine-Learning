# -*- coding: utf-8 -*-

from math import log

import operator
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] # normally show the chinese character
plt.rcParams['axes.unicode_minus']=False # normally show the character '-'


dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
labels = ['no surfacing','flippers']


# calculate the Shannon entropy of the specific dataset according to the final label
def calcShannonEnt(dataSet):
    '''
    :param dataSet: the dataSet has m elements, each element has n features and the label,
    the format of dataset is shown below:
        [[value_1_1,value_1_2,...,value_1_n,label_1],
        [value_2_1,value_2_2,...,value_2_n,label_2],
        ...
        [value_m_1,value_m_2,...,value_m_n,label_m]]
    :return: the float value of the entropy
    '''
    # calculate the total number of the entries
    numEntries = len(dataSet)
    labelCounts = {}

    # get the statistic count of each label
    for featVec in dataSet: 
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # calculate the shannon entropy
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt

# extract the sub dataset according to the value in specific feature
def splitDataSet(dataSet, axis, value):
    '''
    :param dataSet: the input dataset
    :param axis: the feature index
    :param value: the value in the feature with index axis
    :return: return the sub dataset we extract
    '''
    retDataSet = []
    for featVec in dataSet:
    # extract the features with specfic value
        if featVec[axis] == value:
            # drop out the axis-th column
            reducedFeatVec = featVec[:axis]     
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
 

# get the feature with highest gain
def chooseBestFeatureToSplit(dataSet, algorithm = 'ID3'):
    # features number, subtract the label
    numFeatures = len(dataSet[0]) - 1

    # the original entropy before split the dataset
    baseEntropy = calcShannonEnt(dataSet)

    bestInfoGain = 0.0; bestFeature = -1

    for i in range(numFeatures): 
        # get the values in specific column
        featList = [example[i] for example in dataSet]
        # get the union of values in specific column
        uniqueVals = set(featList)       
        newEntropy = 0.0
        IV = 0.0 # for C4.5 algorithm
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            IV -= prob * log(prob,2)
            # the entropy if we use the i-th feature as the criterion to split dataset
            newEntropy += prob * calcShannonEnt(subDataSet)  
    # calculate the information gain
        infoGain = baseEntropy - newEntropy
        if algorithm == 'C45':
            infoGain /= IV

    # chose the i-th feature with highest information gain
        if (infoGain > bestInfoGain):       
            bestInfoGain = infoGain         
            bestFeature = i
    return bestFeature 

# if we transverse all the features, but there are still many classes left,
# we chose the feature with largest count as the final answer
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] # the difference with py3.x


# traverse the whole data set util there is no feature left
def createTree(dataSet,labels,algorithm = 'ID3'):
    '''
    :param dataSet: the dataSet format is same as the dataSet in func calcShannonEnt
    :param labels: the feature names
    :return: a tree with the python dictionary type
    '''
    # get the list of classes in the dataSet
    classList = [example[-1] for example in dataSet]


    # *************** The condition of the end of the recursive call **************
    # if all the data in dataSet are belong to one class, stop creating the tree
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # if there is no feature left and still multiple classes in the dataSet, we chose the
    # class with largest count as the answer
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # *************** The condition of the end of the recursive call **************

    # chose the feature with highest information gain
    bestFeat = chooseBestFeatureToSplit(dataSet,algorithm)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}

    # delete the label has been used
    del (labels[bestFeat])


    # get all the values belong to the bestFeature
    featValues = [example[bestFeat] for example in dataSet]
    # get the union of the featValues
    uniqueVals = set(featValues)

    for value in uniqueVals:
        # copy the labels, not reference
        subLabels = labels[:]
        # recursively call the createTree func
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# get the leaf numbers, used to plot the tree
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

# get tree depth, used to plot the tree
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

# decisionNode，leafNode，arrow_args used to define non-leaf node, leaf node and arrow type respectively
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# plot the nodes and arrows
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '''
    :param nodeTxt: the text used to show
    :param centerPt: the center point coordinate of the text
    :param parentPt: the coordinate pointing to the text
    :param nodeType: the node type
    '''
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )


# plot the text between the parent and child nodes
def plotMidText(cntrPt, parentPt, txtString):
    # define the coordinate
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center")
    
def plotTree(myTree, parentPt, nodeTxt):
    # calculate the height of the tree
    numLeafs = getNumLeafs(myTree)  
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    # the coordinate of a node's center
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    # the text between parent and child nodes
    plotMidText(cntrPt, parentPt, nodeTxt)
    # plot a node
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    # plot the tree according to the dictionary value
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict': 
            plotTree(secondDict[key],cntrPt,str(key))        
        else: 
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD



def createPlot(myTree):
    '''
    :param myTree: a built tree with dictionary type
    :return:
    '''

    # define a figure, and let the color of the background to be white
    fig = plt.figure(1, facecolor='white')
    # clear the figure
    fig.clf()

    # define the plot, 111 means there is only one image
    # frame on or not(the axis)
    createPlot.ax1 = plt.subplot(111, frameon=False)


    # global variable, Save the width and depth of the tree for easy layout of node locations
    plotTree.totalW = float(getNumLeafs(myTree))
    plotTree.totalD = float(getTreeDepth(myTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0

    # plot the tree
    plotTree(myTree, (0.5,1.0), '')
    plt.xticks([])
    plt.yticks([])
    plt.show()

import pandas as pd
import numpy as np

def readCsvToBuildTree(filename):
    df = pd.read_csv(filename, encoding="gbk")

    npData = np.array(df)
    listData = npData.tolist()

    labels = list(df.columns.values)
    return listData,labels[:-1]

def predict(myTree,Data,features_list):

    while(type(myTree).__name__=='dict'):
        node_value = list(myTree.keys())[0]
        index_num = features_list.index(node_value)
        feature_val = Data[index_num]
        myTree = myTree[node_value][feature_val]
    return myTree




trainData,labels = readCsvToBuildTree('motorbike.csv')
tree=createTree(trainData,labels.copy(),'C45')
# createPlot(tree)

testData = ['21…50','high','USA','']
predict_val = predict(tree,testData,labels)
print(predict_val)
