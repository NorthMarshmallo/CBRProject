import math
import numpy as np
import pandas as pd
#library for parsing microarray expression data files
import GEOparse
import random

from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from scipy.spatial.distance import euclidean, cityblock, chebyshev, canberra, jaccard, correlation
import time

from sklearn.preprocessing import StandardScaler


class ExpressionCase(object):   #creation of class of cases by using terminology from MicroCBR
    def __init__(self, idOfPatient, s, cclass):  #constructor
        self.ID = idOfPatient   #id of patient
        self.S = s   #list with levels of genes's expression
        self.caseclass = cclass   #class of case

def canberra_distance(s1, s2):
    return np.sum(np.divide(np.abs(s1 - s2), np.abs(s1) + np.abs(s2), out = np.zeros_like(np.abs(s1 - s2)),
                            where = (np.abs(s1) + np.abs(s2)) != 0))


def jaccard_distance(s1, s2):
    return np.sum((s1 - s2) ** 2) / (np.sum(s1** 2) +  np.sum(s2 ** 2) -  np.sum(s1 * s2))

def correlation_distance(s1, s2):
    return 0.5 * (1 - np.corrcoef(s1, s2)[0][1]) #corrcoef gives us correlation matrices that shows correlation s1 to s1, s1 to s2, s2 to s1, s2 to s2, we take only s1 to s2 (same as s2 to s1)

def hassanat_distance(s1, s2):
    return np.sum(1 - (1 + np.minimum(s1, s2)) / (1 + np.maximum(s1, s2))) # using only a part of equation system, because xi and yi >=0 
 
def bhattacharyya_distance(s1, s2):
    return - math.log(np.sum((s1 * s2) ** 0.5))

def fractal_2_distance(s1, s2):
    return np.sum(np.abs(s1 - s2) ** 0.5) ** 2 

def fractal_10_distance(s1, s2):
    return np.sum(np.abs(s1 - s2) ** 0.2) ** 5

def text_distance(s1, s2):

    n = s1.size
    s = np.count_nonzero(s1)
    t = np.count_nonzero(s2)

    D = np.sum(np.divide(np.abs(s1 - s2), 0.5 * (s1 + s2), out = np.zeros_like(s1),
                                           where = (s1 + s2) != 0))
    return 1 / (0.5 * (s / n + t * s / (t * s + D)))

def z_distance(s1, s2):

    n = len(s1)
    a = s1[:n//2]
    b = s2[:n//2]
    centre_a = s1[n//2 + 1:]
    centre_b = s2[n//2 + 1:]

    return correlation_distance(a, b) + 1.8 * correlation_distance(centre_a, centre_b)

def z_cityblock_distance(s1, s2):

    n = len(s1)
    a = s1[:n//2]
    b = s2[:n//2]
    centre_a = s1[n//2 + 1:]
    centre_b = s2[n//2 + 1:]

    return cityblock(a, b) + 1.8 * cityblock(centre_a, centre_b)


distances = np.array([text_distance, fractal_2_distance, fractal_10_distance, "euclidean",
                      "cityblock", "chebyshev", canberra_distance, jaccard_distance,
                      correlation_distance, hassanat_distance, bhattacharyya_distance])


def get_sample80336(name, df2, meta):
    sample_title = meta.gsms[name].metadata['title'][0].split('-')
    sample_class = sample_title[0]
    if sample_class == 'control':
        column = 'C_' + sample_title[1]
    else:
        column = 'BD_' + sample_title[1]
    sample = ExpressionCase(column, df2[column].tolist(), sample_class)
    return [sample_class, sample]

def get_sample183947(name, df2, meta):
    sample_class = meta.gsms[name].metadata['characteristics_ch1'][0]
    column = meta.gsms[name].metadata['description']
    sample = ExpressionCase(column, df2[column].values.transpose().tolist()[0], sample_class)
    return [sample_class, sample]

def get_data_rna(meta_path, path, get_sample_func, file_sep = None,
    drop_columns = None, form = None): #form = 'standart'(-1, 1), 'normal'(0, 1)

    meta = GEOparse.get_GEO(geo = meta_path)
    gse = pd.read_csv(path, sep = file_sep)
    print(gse)

    # looking what the file contains
    #with pd.option_context('display.max_rows', 500000, 'display.max_columns', 50000):
     #   for name, extra in gse.gsms.items():
      #      name = name.strip('\n')
       #     print(name)
        #    print(gse.gsms[name].metadata['characteristics_ch1'][0])
         #   print(type(gse.gsms[name].metadata['description']))

    # taking table of expression profiles of samples (columns - genes ID, genes identifier, ID of samples; rows - genes and their expression levels in each samples)
    df = gse.applymap(lambda x: 0.0 if pd.isna(x) else x)

    # dict where keys are names of classes and values are arrays with samples of that classes
    classes = {}

    if drop_columns != None:
        df2 = df.drop(drop_columns, axis=1)
    else:
        df2 = df

    if form == 'normal':
        df2 = df2.sub(df2.min(1), axis=0).div(df2.max(1) - df2.min(1), axis=0).dropna()
    elif form == 'standart':
        df2 = df2.sub(df2.mean(1), axis=0).div(df2.std(1), axis=0).dropna()
    print('1', df2)
    # print(df['GSM907858'])

    for name, extra in meta.gsms.items():
        # return name of disease
        sample_class, sample = get_sample_func(name, df2, meta)
        try:
            classes[sample_class].append(sample)
        except:
            classes[sample_class] = [sample]

    return classes

def get_data_microarray(path, form = None): #form = 'standart'(-1, 1), 'normal'(0, 1)
    
    gse = GEOparse.get_GEO(filepath = path)
    #parsed_gse = open("parsed_gse.txt", "w+")
    
    #looking what the file contains
    #with pd.option_context('display.max_rows', 500000, 'display.max_columns', 50000):
     #   parsed_gse.write(str(gse.table))
      #  parsed_gse.write('\n')
       # parsed_gse.write(str(gse.columns))
        #parsed_gse.close()

    #taking table of expression profiles of samples (columns - genes ID, genes identifier, ID of samples; rows - genes and their expression levels in each samples)   
    #df = (gse.table).applymap(lambda x: 0.0 if pd.isna(x) else x)
    df = gse.table

    #taking table with ID of samples as indexes and names of diseases in rows
    datasetInfo = gse.columns
    #dict where keys are names of classes and values are arrays with samples of that classes
    classes = {}
    
    df2 = df.filter(regex = ("GSM.*"), axis = 1)
    if form == 'normal':
        df2 = df2.sub(df2.min(1), axis = 0).div(df2.max(1) - df2.min(1), axis = 0)
    elif form == 'standart':
        df2 = df2.sub(df2.mean(1), axis = 0).div(df2.std(1), axis = 0)
    
    for column in df.columns:

        if column[:3] == 'GSM':
            #return name of disease
            sample_class = datasetInfo.loc[datasetInfo.index == column]['disease state'].tolist()[0]
            sample = ExpressionCase(column, df2[column].tolist(), sample_class)
            try:
                classes[sample_class].append(sample)
            except:
                classes[sample_class] = [sample]
    
    return classes

def f1_complexity_measure(classes):
    
    means = []
    stdevs = []
    
    for diagnos in classes.keys():
        class_samples = classes[diagnos]
        means.append(np.mean([x.S for x in class_samples], axis = 0)) #counting mean and standart diviation in each class, so we can know f1 later
        stdevs.append(np.std([x.S for x in class_samples], axis = 0))

    f1 = np.max(np.nan_to_num((means[0] - means[1])**2 / (stdevs[0] + stdevs[1])))
    
    return f1
        

def train_test_split(classes):
    
    train_data = []
    test_data = []    
    pred_train = []
    pred_test = []
    
    #dividing data into test and train where approximately 80% is train data 
    for diagnos in classes.keys():
        class_samples = classes[diagnos]
        random.shuffle(class_samples)
        num_samples_in_class = len(class_samples)
        for i in range(num_samples_in_class):
            #after random shuffling, first 70% of data goes to train and over 30% goes to  test array
            if i < (num_samples_in_class - 1) * 0.7:
                train_data.append(class_samples[i].S)
                pred_train.append(diagnos)
            else:
                test_data.append(class_samples[i].S)
                pred_test.append(diagnos)  
    
    return [train_data, test_data, pred_train, pred_test]
    

def CBR_Realise(train_test_set, distance, class_names, threshould = 0.5): #dict where keys are names of classes and values are arrays with samples of that classes
    
    train_data, test_data, pred_train, pred_test = train_test_set

    #getting classification for test samples by using different metrics 

    knn = KNeighborsClassifier(n_neighbors = 7, metric = distance, weights = "uniform")
    knn.fit(train_data, pred_train)
 
    #in our case it's better when recall for A class is higher, this parts helps to set A class with higher probability
    knn_results_recall = [ class_names[0] if item else class_names[1] for item in (knn.predict_proba(test_data)[:,0] >= threshould)]
    
    metr_res_recall = classification_report(pred_test, knn_results_recall, output_dict = True,
                                            zero_division = False)
    
    #accuracy and recall for classification of test data
    metr_step_scores_recall = [metr_res_recall['accuracy'], metr_res_recall[class_names[0]]['recall'],
                               metr_res_recall[class_names[1]]['recall']]
    
    return [metr_step_scores_recall]

def append_centres(train_test_set, class_names):

    train_data, test_data, pred_train, pred_test = train_test_set

    overall_mean = np.mean(train_data, axis = 0)
    a_mean = []
    na_mean = []

    train_len = len(pred_train)

    for i in range(train_len):
        if pred_train[i] == class_names[0]:
            a_mean.append(train_data[i])
        if pred_train[i] == class_names[1]:
            na_mean.append(train_data[i])

    a_mean = np.mean(a_mean, axis = 0)
    na_mean = np.mean(na_mean, axis = 0)

    for sample in test_data:
        sample += overall_mean
    for i in range(train_len):
        if pred_train[i] == class_names[0]:
            train_data[i] += a_mean
        if pred_train[i] == class_names[1]:
            train_data[i] += na_mean

def n1_complexity_measure(classes,distance):

    samples_1 = list(classes.values())
    samples = samples_1[0] + samples_1[1]
    n = len(samples)
    G = []
    for i in range(n):
        g_row = []
        for j in range(n):
            g_row.append(distance(np.array(samples[i].S), np.array(samples[j].S)))
        G.append(g_row)

    INF = 9999999
    # number of vertices in graph
    N = n
    diff_class_count = 0

    selected_node = [0] * n

    no_edge = 0

    selected_node[0] = True

    checked = []
    
    while (no_edge < N - 1):

        minimum = INF
        a = 0
        b = 0
        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if ((not selected_node[n]) and G[m][n]):
                        # not in selected and there is an edge
                        if minimum > G[m][n]:
                            minimum = G[m][n]
                            a = m
                            b = n
                            if samples[n].caseclass != samples[m].caseclass:
                                if samples[n] not in checked:
                                    diff_class_count += 1
                                    checked.append(samples[n])
                                if samples[m] not in checked:
                                    diff_class_count += 1
                                    checked.append(samples[m])
        selected_node[b] = True
        no_edge += 1
    return diff_class_count / N

def n2_complexity_measure(classes,distance):

    samples_1 = list(classes.values())
    samples = samples_1[0] + samples_1[1]
    n = len(samples)

    inside_all = []
    outside_all = []
    for i in range(n):
        inside = []
        outside = []
        for j in range(n):
            if samples[i].caseclass != samples[j].caseclass:
                outside.append(distance(np.array(samples[i].S), np.array(samples[j].S)))
            elif i != j:
                inside.append(distance(np.array(samples[i].S), np.array(samples[j].S)))
        inside_all.append(min(inside))
        outside_all.append(min(outside))

    in_mean = np.mean(inside_all)
    out_mean = np.mean(outside_all)
    n2 = in_mean / out_mean

    return n2

def n3_complexity_measure(classes, distance):

    samples_1 = list(classes.values())
    samples = samples_1[0] + samples_1[1]
    X = [sample.S for sample in samples]
    y = [sample.caseclass for sample in samples]
    n = len(X)
    knn = KNeighborsClassifier(n_neighbors=1, metric=distance, weights="uniform")
    scores = cross_val_score(knn, X, y, cv = LeaveOneOut())
    return np.count_nonzero(scores == 0) / n


if __name__=="__main__":

    #classes = get_data_rna('GSE80336', './GSE80336_Counts.txt.gz', get_sample80336, '\t', ['Ensembl_ID', 'GeneSymbol',
                                                                  # 'Biotype', 'Chromosome'])

    #classes = get_data_rna('GSE183947', './GSE183947_fpkm.csv.gz', get_sample183947, None, ['Unnamed: 0'])
    classes = get_data_microarray("./GDS4758_full.soft.gz")

    class_names = list(classes.keys())
    print(class_names)
    print(len(classes[class_names[0]]), len(classes[class_names[1]]))

    n = 4
    #prepating same train test sets for all distances
    train_test_sets = []
    for i in range(n):
        train_test_sets.append(train_test_split(classes))

    all_res = []
    times = []

    for distance in distances:
        metr_res = []
        time_start = time.time()
        for i in range(n):
            #gives accuracy and recall for classification on random test data for given metric
            #0.423
            metr_step_scores = CBR_Realise(train_test_sets[i], distance, class_names, 0.423)
            #keeps results from all n tests for given metric
            metr_res.append(metr_step_scores)
        #keeps results for all metric
        all_res.append(metr_res)
        times.append((time.time() - time_start) / n)

    for distance in [z_distance, z_cityblock_distance]:
        metr_res = []
        #special for reachable distance
        time_start = time.time()
        for i in range(n):
            #sending vectors with their class centre info
            append_centres(train_test_sets[i], class_names)
            #0.35
            metr_step_scores = CBR_Realise(train_test_sets[i], distance, class_names,0.35)
            # keeps results from all n tests for given metric
            metr_res.append(metr_step_scores)
        all_res.append(metr_res)
        times.append((time.time() - time_start) / n)

print() 
# counting percentage of winning by accuracy for each metric (metric wins on one of n splits if its value is max of all distances for this split)
win_metr = np.array(all_res).argmax(0)
print(win_metr)

#count statistics for reqular knn and then for one special for A recall  
for i in range(1):
    unique, counts = np.unique(win_metr[:, i][:, 0], return_counts = True)
    print(dict(zip(unique, counts / n)))
    # counting average accuracy, recall for alz and recall for non-alz across n same train/test splits

print(np.sum(all_res, axis = 1) / n)
print()
print(times)
print("Dataset f1 complexity measure: ", f1_complexity_measure(classes))
