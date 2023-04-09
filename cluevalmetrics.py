"""
Clustering evaluation metrics for named entity disambiguation

(1) cluster-f: cluster-f precision/recall/f1
(2) k-metric: k-metric precision/recall/f1
(3) split-lump: splitting & lumping error precision/recall/f1
(4) pairwise-f: paired precision/recall/f1    
(5) b-cubed: b3 precision/recall/f1  

For more details on clustering evaluation metrics, see a paper below
Kim, J. (2019). A fast and integrative algorithm for clustering performance evaluation
    in author name disambiguation. Scientometrics, 120(2), 661-681. 
	
"""

import math 
import numpy as np


def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred

def clusterf_precision_recall_fscore(labels_true, labels_pred):
    """Compute the cluster-f of precision, recall and F-score.
    
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.

    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f1-score: calculated f1-score
    
    Reference
    ---------
    Kim, J. (2019). A fast and integrative algorithm for clustering performance evaluation
    in author name disambiguation. Scientometrics, 120(2), 661-681.
    """
    
    # check that labels_* are 1d arrays and have the same size
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # check that input given is not the empty set
    if labels_true.shape == (0, ):
        raise ValueError("input labels must not be empty.")

    n_samples = len(labels_true)
    true_clusters = {} 
    pred_clusters = {}  
    
	# create a list of clusters
    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    true_keys = list(true_clusters.keys())
    true_keys.sort()
    truth = [list(true_clusters[i]) for i in true_keys]

    pred_keys = list(pred_clusters.keys())
    pred_keys.sort()
    predicted = [list(pred_clusters[i]) for i in pred_keys]
    
	# compute clustering evaluation metric
    cSize = {}
    cMatch = 0
    pIndex = {}
	
    for i, pred_i in zip(range(len(predicted)),predicted):
        for p in pred_i:
            pIndex[p] = i + 1

        cSize[i + 1] = len(pred_i)        

    for true_j in truth:
        tMap = {}

        for t in true_j:
            if not pIndex[t] in tMap.keys():
                tMap[pIndex[t]] = 0
            tMap[pIndex[t]] = tMap[pIndex[t]] + 1

        for key, value in sorted(tMap.items(), key = lambda kv: kv[0]):
            if value == len(true_j) and cSize[key] == len(true_j):
                cMatch += 1   
            
    recall = cMatch/len(truth)
    precision = cMatch/len(predicted)

    try:
        f_score = 2*recall*precision/(recall + precision)
    except ZeroDivisionError:
        f_score = 1.0	

    return (precision, recall, f_score)

def kmetric_precision_recall_fscore(labels_true, labels_pred):
    """Compute the k-metric of precision, recall and F-score.
    
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.

    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f1-score: calculated f1-score
    
    Reference
    ---------
    Kim, J. (2019). A fast and integrative algorithm for clustering performance evaluation
    in author name disambiguation. Scientometrics, 120(2), 661-681.
    """
    
    # check that labels_* are 1d arrays and have the same size
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # check that input given is not the empty set
    if labels_true.shape == (0, ):
        raise ValueError("input labels must not be empty.")

    n_samples = len(labels_true)
    true_clusters = {}  
    pred_clusters = {}  
    
	# create a list of clusters
    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    true_keys = list(true_clusters.keys())
    true_keys.sort()
    truth = [list(true_clusters[i]) for i in true_keys]

    pred_keys = list(pred_clusters.keys())
    pred_keys.sort()
    predicted = [list(pred_clusters[i]) for i in pred_keys]

    # compute clustering evaluation metric
    cSize = {}
    cMatch = 0
    aapSum = 0 
    acpSum = 0
    pIndex = {}
	
    for i, pred_i in zip(range(len(predicted)),predicted):
        for p in pred_i:
            pIndex[p] = i + 1
        cSize[i + 1] = len(pred_i)       
	
    instSum = 0
    for true_j in truth:
        instSum += len(true_j)
        tMap = {}

        for t in true_j:
            if not pIndex[t] in tMap.keys():
                tMap[pIndex[t]] = 0
            tMap[pIndex[t]] = tMap[pIndex[t]] + 1

        for key, value in sorted(tMap.items(), key = lambda kv: kv[0]):
            if value == len(true_j) and cSize[key] == len(true_j):
                cMatch += 1   
            aapSum += pow(value,2)/len(true_j)
            acpSum += pow(value,2)/cSize[key]
	
    recall = aapSum/instSum
    precision = acpSum/instSum

    try:
        f_score = math.sqrt(recall*precision)
    except ZeroDivisionError:
        f_score = 1.0	

    return (precision, recall, f_score)

def split_lump_error_precision_recall_fscore(labels_true, labels_pred):
    """Compute the splitting & lumping error with precision, recall and F-score.
    
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.

    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f1-score: calculated f1-score
    
    Reference
    ---------
    Kim, J. (2019). A fast and integrative algorithm for clustering performance evaluation
    in author name disambiguation. Scientometrics, 120(2), 661-681.
    """
    
    # check that labels_* are 1d arrays and have the same size
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # check that input given is not the empty set
    if labels_true.shape == (0, ):
        raise ValueError("input labels must not be empty.")

    n_samples = len(labels_true)
    true_clusters = {}  
    pred_clusters = {}  
    
	# create a list of clusters
    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    true_keys = list(true_clusters.keys())
    true_keys.sort()
    truth = [list(true_clusters[i]) for i in true_keys]

    pred_keys = list(pred_clusters.keys())
    pred_keys.sort()
    predicted = [list(pred_clusters[i]) for i in pred_keys]
    
	# compute clustering evaluation metric
    cSize = {}
    spSum = 0
    lmSum = 0
    instTrSum = 0
    instPrSum = 0
    pIndex = {}
	
    for i, pred_i in zip(range(len(predicted)),predicted):
        for p in pred_i:
            pIndex[p] = i + 1

        cSize[i + 1] = len(pred_i)         # hash of a cluster P_i and its size
		
    for true_j in truth:
        tMap = {}
        maxKey = 0
        maxValue = 0

        for t in true_j:
            if not pIndex[t] in tMap.keys():
                tMap[pIndex[t]] = 0
            tMap[pIndex[t]] = tMap[pIndex[t]] + 1

        for key, value in sorted(tMap.items(), key = lambda kv: kv[0]):
            if value > maxValue:
                maxValue = value
                maxKey = key
            instTrSum += len(true_j) 
            instPrSum += cSize[maxKey] 
            spSum += (len(true_j) - maxValue)  
            lmSum += (cSize[maxKey] - maxValue) 
	
    SE = spSum/instTrSum
    LE = lmSum/instPrSum
  
    recall = 1 - SE
    precision = 1 - LE

    try:
        f_score = (2*recall*precision)/(recall + precision)
    except ZeroDivisionError:
        f_score = 1.0

    return (precision, recall, f_score)

def pairwisef_precision_recall_fscore(labels_true, labels_pred):
    """Compute the pairwise-f of precision, recall and F-score.
    
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.

    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f1-score: calculated f1-score
    
    Reference
    ---------
    Kim, J. (2019). A fast and integrative algorithm for clustering performance evaluation
    in author name disambiguation. Scientometrics, 120(2), 661-681.
    """
    
    # check that labels_* are 1d arrays and have the same size
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # check that input given is not the empty set
    if labels_true.shape == (0, ):
        raise ValueError("input labels must not be empty.")
    
    n_samples = len(labels_true)
    true_clusters = {}  
    pred_clusters = {}  
    
	# create a list of clusters
    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)
    
    true_keys = list(true_clusters.keys())
    true_keys.sort()
    truth = [list(true_clusters[i]) for i in true_keys]

    pred_keys = list(pred_clusters.keys())
    pred_keys.sort()
    predicted = [list(pred_clusters[i]) for i in pred_keys]
    
	## compute clustering evaluation metric
    pairPrSum = 0
    pairTrSum = 0
    pairIntSum = 0
    pIndex = {}
	  
    for i, pred_i in zip(range(len(predicted)),predicted):
        for p in pred_i:
            pIndex[p] = i + 1
        pairPrSum += len(pred_i)*(len(pred_i) - 1)/2
		

    for true_j in truth:
        pairTrSum += len(true_j)*(len(true_j) - 1)/2
        tMap = {}

        for t in true_j:
            if not pIndex[t] in tMap.keys():
                tMap[pIndex[t]] = 0
            tMap[pIndex[t]] = tMap[pIndex[t]] + 1

        for key, value in sorted(tMap.items(), key = lambda kv: kv[0]):
            pairIntSum += value*(value - 1)/2
            
	
    try:
        recall = pairIntSum/pairTrSum
    except ZeroDivisionError:
        recall = 1.0
    
    try:
        precision = pairIntSum/pairPrSum
    except ZeroDivisionError:
        precision = 1.0

    try:
        f_score = (2*recall*precision)/(recall+precision)
    except ZeroDivisionError:
        f_score = 1.0

    return (precision, recall, f_score)

def bcubed_precision_recall_fscore(labels_true, labels_pred):
    """Compute the b-cubed metric of precision, recall and F-score.
    
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.

    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f1-score: calculated f1-score
    
    Reference
    ---------
    Kim, J. (2019). A fast and integrative algorithm for clustering performance evaluation
    in author name disambiguation. Scientometrics, 120(2), 661-681.
    """
    
    # check that labels_* are 1d arrays and have the same size
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # check that input given is not the empty set
    if labels_true.shape == (0, ):
        raise ValueError("input labels must not be empty.")
    
	# create a list of clusters
    n_samples = len(labels_true)
    true_clusters = {}  
    pred_clusters = {}  

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)
    
    true_keys = list(true_clusters.keys())
    true_keys.sort()
    truth = [list(true_clusters[i]) for i in true_keys]

    pred_keys = list(pred_clusters.keys())
    pred_keys.sort()
    predicted = [list(pred_clusters[i]) for i in pred_keys]
    
    ## compute clustering evaluation metric
    cSize = {}
    cMatch = 0
    aapSum = 0 
    acpSum = 0
    pIndex = {}
	
    for i, pred_i in zip(range(len(predicted)),predicted):
        for p in pred_i:
            pIndex[p] = i + 1

        cSize[i + 1] = len(pred_i) # hash of a cluster P_i and its size
		
    instSum = 0	
    for true_j in truth:
        instSum += len(true_j)
        tMap = {}

        for t in true_j:
            if not pIndex[t] in tMap.keys():
                tMap[pIndex[t]] = 0
            tMap[pIndex[t]] = tMap[pIndex[t]] + 1

        for key, value in sorted(tMap.items(), key = lambda kv: kv[0]):
            if value == len(true_j) and cSize[key] == len(true_j):
                cMatch += 1     
            aapSum += pow(value,2)/len(true_j)
            acpSum += pow(value,2)/cSize[key]
	
    recall = aapSum/instSum
    precision = acpSum/instSum
  
    try:
        f_score = 2*recall*precision/(recall + precision)
    except ZeroDivisionError:
        f_score = 1.0

    return (precision, recall, f_score)
### The end of line ###