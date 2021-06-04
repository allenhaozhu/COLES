from __future__ import print_function
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from metrics import clustering_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import scipy.sparse as sp

def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(solver='liblinear', multi_class='ovr')
    log.fit(train_embeds, train_labels)
    predict = (log.predict(test_embeds)).tolist()
    accuracy = accuracy_score(test_labels, predict)
    print("Test Accuracy:", accuracy)
    return accuracy

def classify(embeds, dataset, per_class):

    label_file = open("data/{}{}".format(dataset,"_labels.txt"), 'r')
    label_text = label_file.readlines()
    labels = []
    for line in label_text:
        if line.strip('\n'):
            line = line.strip('\n').split(' ')
            labels.append(int(line[1]))
    label_file.close()
    labels = np.array(labels)
    train_file = open("data/{}/{}/train_text.txt".format(dataset, per_class), 'r')
    train_text = train_file.readlines()
    train_file.close()
    test_file = open( "data/{}/{}/test_text.txt".format(dataset, per_class), 'r')
    test_text = test_file.readlines()
    test_file.close()
    ave = []
    for k in range(50):
        train_ids = eval(train_text[k])
        test_ids = eval(test_text[k])
        train_labels = [labels[i] for i in train_ids]
        test_labels = [labels[i] for i in test_ids]
        train_embeds = embeds[[id for id in train_ids]]
        test_embeds = embeds[[id for id in test_ids]]
        acc = run_regression(train_embeds, train_labels, test_embeds, test_labels)
        ave.append(acc)
    print(np.mean(ave)*100)
    print(np.std(ave)*100)


def clustering(embeds, dataset):

    label_file = open("data/{}{}".format(dataset,"_labels.txt"), 'r')
    label_text = label_file.readlines()
    labels = []
    for line in label_text:
        if line.strip('\n'):
            line = line.strip('\n').split(' ')
            labels.append(int(line[1]))
    label_file.close()
    labels = np.array(labels)
    rep = 10
    # u, s, v = sp.linalg.svds(embeds, k=32, which='LM')
    # u = normalize(embeds.dot(v.T))
    u = embeds
    k = len(np.unique(labels))
    ac = np.zeros(rep)
    nm = np.zeros(rep)
    f1 = np.zeros(rep)
    for i in range(rep):
        kmeans = KMeans(n_clusters=k).fit(u)
        predict_labels = kmeans.predict(u)
        #intraD[i] = square_dist(predict_labels, u)
        # intraD[i] = dist(predict_labels, feature)
        cm = clustering_metrics(labels, predict_labels)
        ac[i], nm[i], f1[i] = cm.evaluationClusterModelFromLabel()

    print(np.mean(ac))
    print(np.std(ac))
    print(np.mean(nm))
    print(np.std(nm))
    print(np.mean(f1))
    print(np.std(f1))
