import math
import numpy as np
import pandas as pd
#library for parsing microarray expression data files
import GEOparse
import time

class MicroarrayCase(object):   #creation of class of cases by using terminology from MicroCBR
    def __init__(self, idOfPatient, s, cclass):  #constructor
        self.ID=idOfPatient   #id of patient
        self.S=s   #list with levels of genes's expression
        self.caseclass=cclass   #class of case

def eucldistance(self, case2):  #distance carried by using euclid metric
    s1=self.S
    s2=case2.S
    distance=0 
    for i in range(len(s1)):
        distance+=(s1[i]-s2[i])**2
    return math.sqrt(distance)

def manhdistance(self,case2):
    s1=self.S
    s2=case2.S
    distance=0 
    for i in range(len(s1)):
        distance+=abs(s1[i]-s2[i])
    return math.sqrt(distance)

def normdistance(self,case2):
    s1=self.S
    s2=case2.S
    distance=0 
    for i in range(len(s1)):
        if distance<abs(s1[i]-s2[i]):
            distance=abs(s1[i]-s2[i])
    return distance

distances=np.array([eucldistance,manhdistance,normdistance])

#trainData is array of train cases, testData is array of test cases
def classifyKNN (trainData, testData, k, classes, distances):
    #u=time.time()
    #!!if k>len(trainData):
    testLabels = []
    number_of_train_samples = len(trainData)
    for dist in distances:
        testL = []
        for testPoint in testData:
            #Claculate distances between test point and all of the train points
            testDist = [ [dist(testPoint, trainData[i]), trainData[i].caseclass] for i in range(number_of_train_samples)]
            #u1=time.time()
            #print(u1-u,'dist')
            #How many points of each class among nearest K
            stat = {}
            for i in classes.keys():
                stat[i] = 0
            pointClass = 0
            #Assign a class with the most number of occurences among K nearest neighbours
            for d in sorted(testDist)[0:k]:
                stat[d[1]] += 1
                tmp = stat[d[1]]
                if tmp > pointClass:
                    pointClass = tmp
                    nearestClass = d[1]
            #print(testPoint.caseclass, stat)
            #u2=time.time()
            #print(u2-u1,'class')
            #u=u2
            testL.append(nearestClass)
        testLabels.append(testL)
    return testLabels

def choose_best_metric(testData,kNNresults):
    indBestDistance = 0
    maxOfPoints = 0
    nd = len(kNNresults)
    ntd = len(testData)
    #array for counting how many of test points were classified correctly
    metricPoints = [0 for i in range(nd)]
    for i in range(ntd):
        for j in range(nd):
            if kNNresults[j][i] == testData[i].caseclass:
                metricPoints[j] += 1
                if metricPoints[j] > maxOfPoints:
                    indBestDistance = j
    print(metricPoints)
    return distances[indBestDistance]

def CBR_Realise(path):
    trainData = []
    testData = []
    gse = GEOparse.get_GEO(filepath=path)
    parsed_gse = open("parsed_gse.txt", "w+")
    #looking what the file contains
    #with pd.option_context('display.max_rows', 500000, 'display.max_columns', 50000):
        #parsed_gse.write(str(gse.table))
        #parsed_gse.write('\n')
        #parsed_gse.write(str(gse.columns))
        #parsed_gse.close()
        
    #taking table of expression profiles of samples (columns - genes ID, genes identifier, ID of samples; rows - genes and their expression levels in each samples)   
    df = (gse.table).applymap(lambda x: 0.0 if pd.isna(x) else x)
    #taking table with ID of samples as indexes and names of diseases in rows
    datasetInfo = gse.columns
    #dict where keys are names of classes and values are arrays with samples of that classes
    classes={}
    
    for column in df.columns:
        if column[:3]=='GSM':
            #return name of disease
            sample_class = datasetInfo.loc[datasetInfo.index==column]['disease state'].tolist()[0]
            sample = MicroarrayCase(column,df[column].tolist(),sample_class)
            try:
                classes[sample_class].append(sample)
            except:
                classes[sample_class]=[sample]
                
    #dividing data into test and train where approximately 80% is train data 
    for diagnos in classes.keys():
        tmp=classes[diagnos]
        numSamplesInClass=len(tmp)
        for i in range(numSamplesInClass):
            if i<(numSamplesInClass-1)*0.8:
                trainData.append(tmp[i])
            else:
                testData.append(tmp[i])
    
    #getting classification for test samples by using different metrics 
    kNNresults=classifyKNN(trainData, testData, 8, classes, distances)            
    print(kNNresults)
    #choosing best metric
    print(choose_best_metric(testData, kNNresults))

if __name__=="__main__":
    CBR_Realise("./GDS5306.soft.gz")