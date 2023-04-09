######################################################################################
# ANDez is an open-source framework that integrates the workflows of several high-performing machine learning methods 
# for classification and clustering in author name disambiguation.
# ANDez was developed under a grant from the National Science Foundation 
# (NSF NCSES Award # 1917663: Creating a Data Quality Control Framework for Producing New Personnel-Based S&E Indicators) 
# and its supplementary fund program, Research Experiences for Undergraduates (REU).
#
# Author:
# 1. Jinseok Kim (Ph.D.): Institute for Social Research and School of Information, University of Michian Ann Arbor
# 2. Jenna Kim: School of Information Sciences, University of Illinois at Urbana-Champaign
#
######################################################################################


# NOTE: Run this script with 'procedures.py' and 'modules.py' in the same directory


import time
import uuid
from datetime import timedelta

from procedures import *

''' measure start time '''
start_time = time.time()


##################### set parameters #####################
"""
For details on parameter choices during disambiguation, see a paper below
(1) Kim, J., & Kim, J. (2020). Effect of forename string on author name disambiguation.
    Journal of the Association for Information Science and Technology, 71(7), 839-855
(2) Kim, J., Kim, J., & Owen-Smith, J. (2019). Generating automatically labeled data for
    author name disambiguation: an iterative clustering method. Scientometrics, 118(1), 253-280.
(3) Kim, J., & Owen-Smith, J. (2020). Model Reuse in Machine Learning for Author Name Disambiguation:
    An Exploration of Transfer Learning. IEEE Access, 8, 188378-188389. doi:10.1109/ACCESS.2020.3031112
"""


### 1. Type in file names ###

"""
Input files are required to be prepared in a specific format
(1) signature file: instance id, paper id, author byline position, name string, affiliation, etc. 
(2) record file: paper id, publication year, venue name, author list, title, etc.
                > Author names in the author list are seperated by vertical bar
(3) cluster file: cluster id and instance id list
                > Instance ids in instance id list are separated by vertical bar
Each file is created in .txt and columns are separated by tab
Please see the example files provided with this code set 
"""

train_instance_file = 'signatures_train.txt'
train_cluster_file  = 'clusters_train.txt'
train_record_file   = 'records.txt'

test_instance_file  = 'signatures_test.txt'
test_cluster_file   = 'clusters_test.txt'
test_record_file    = 'records.txt'


### 2. Choose a blocking method ###

"""
Blocking is a step to collate name instances to be compared with each other
The blocking method selected here is applied to both training and test data

Three options are available
(1) first_initial: name instances that have the same surname and first forename initial are compared
        e.g., 'kim, jinseok' vs 'kim, j' > They share 'kim, j'
(2) full_name: name instances that have the same string are compared
        e.g., 'kim, jinseok' vs 'kim, jinseok' > They share 'kim, jinseok'
(3) forename_strip: name instances that have the same surname and n characters of forename are compared
        e.g., 'kim, jinseok' vs 'kim, jin s' > They share 'kim, jin' (if n == 3)

For more details on blocking and 3 options, see the paper below
    Kim, J., & Kim, J. (2020). Effect of forename string on author name disambiguation.
        Journal of the Association for Information Science and Technology, 71(7), 839-855. doi:10.1002/asi.24298
"""

blocking_method = "first_initial"

print("\nBlocking method: '" + blocking_method + "' is selected\n")


### 3. Choose a similarity calculation metric ###

"""
(1) cos: cosine similarity
(2) jac: Jaccard similarity
(3) jrw: Jaro-Winkler similarity

"""

similarity_metric = "cos"      

print( "\nSimilarity calculation metric: '" + similarity_metric + "' is selected\n" )


### 4. Choose one or more classifiers for pairwise similarity comparison ###

"""
(1) GB: Gradient Boosting
(2) RF: Random Forests
(3) LR: Logistic Regression
(4) NB: Naive Bays;
(5) SVM: Support Vector Machine
(6) DT: Decision Tree;

Choice of multiple classifier names available: e.g., classifier_lists = ['LR', 'RF', 'SVM']
URLs for details on each clssifier are available in procedures.py

"""

classifier_name_list = ['LR']


### 5. Which file contains clusters to be used for test? ###

"""
(1) None: no test_cluster_file is provided -> clustering evaluation is unavailable 
(2) Otherwise, 'test_cluster_file'

"""

cluster_file_test = test_cluster_file


### 6. 10-fold cross validation is performed? ###

"""
(1) 1: yes
(2) 0: no

"""

cross_validation = 0


### 7. produce classification results? ###

"""
(1) 1: yes -> classification performance is evaluated on labeled test data
              and classification report is produced for precision, recall, and f1-score
(2) 0: no

"""

conduct_classification = 1   


### 8. Choose a clustering algorithm ###

"""
Clustering is a process of an algorithm to collate name instances into clusters
(1) hier: hierarchical agglomerative clustering
        -> NOTE! you must change below options < clusterer_blocking_on = 1, cluster_count = None >
        -> This process is implemented by the BEARD library for computational efficiency as introduced in
           Louppe, G., Al-Natsheh, H. T., Susik, M., & Maguire, E. J. (2016).
           Ethnicity Sensitive Author Disambiguation Using Semi-supervised Learning.
           Knowledge Engineering and Semantic Web, Kesw 2016, 649, 272-287. 

(2) db: DBSCAN
(3) spectral: spectral
(4) kmeans: K-Means
(5) agg: agglomerative clustering: change below options <clusterer_blocking_on = 0, cluster_count = integer number>

URLs for details on each clssifier are available in modules.py

"""

clustering_algorithm = "hier"

cluster_blocking_on = 1 # 1 for clustering with blocking applied (hierarchical); 0 for other clustering methods

cluster_count = None    # "None" for hierarchical clustering; integer(e.g., 1000) for DBSCAN, spectral, KMeans or agglomerative
    

### 9. If 'hier' is chosen, what is a threshold value? ###

"""
Set a threshold value to filter instance pairs to be put into the same cluster
between 0 and 1
A threshold value is a distance, i.e., 1 - similarity score, between name instances.
E.g., A threshold value of 0.3 is roughly equal to 70 % of probability of name instances
      referring to the same author entity
The lower the threshold value is, the higher the precision score is.

"""

threshold_list = [0.35] # [0] if clustering algorithms other than 'hier' are used
    # <- if various thresholds need to be used, put a starting threshold, an end threshold, and a number of samples in the list
    # e.g., [0.1, 0.3, 5]: this generates a list of thresholds [0.1, 0.15, 0.2, 0.25, 0.3]


### 10. Which clustering evaluation metric do you want to use? ###

"""
(1) cluster-f: cluster-f precision/recall/f1
(2) k-metric: k-metric precision/recall/f1
(3) split-lump: splitting & lumping error precision/recall/f1
(4) pairwise-f: paired precision/recall/f1    
(5) b-cubed: b3 precision/recall/f1  

For more details on clustering evaluation metrics, see a paper below
Kim, J. (2019). A fast and integrative algorithm for clustering performance evaluation
    in author name disambiguation. Scientometrics, 120(2), 661-681. 

"""

clustering_metric = "b-cubed" 


### 11. Would you like to assign a distinct identifier to each cluster? ###

"""
enable_cluster_id = True
enable_cluster_id = False 


The parameter enable_cluster_id controls whether a unique identifier is 
assigned to each cluster within the namespace "550e8400-e29b-41d4-a716-44665544abcd".
This can be useful for tracking individual clusters throughout an analysis. 
To enable cluster ID, set enable_cluster_id to True. The output file includes 
IDs in the first column and cluster lists in the second column with a tab as a delimiter. 
To disable cluster ID, set enable_cluster_id to False.

The namespace used in this script is a UUID (Universally Unique Identifier) 
generated with the value '550e8400-e29b-41d4-a716-44665544abcd'. A UUID is a 128-bit
identifier that is globally unique and can be used to prevent naming conflicts 
between different systems or entities. This namespace is used to create deterministic 
UUIDs using the uuid5() function from the uuid module, which takes a namespace and 
a name as input and generates a UUID based on them. 

"""

enable_cluster_id = True
cluster_id_namespace = uuid.UUID('550e8400-e29b-41d4-a716-44665544abcd')


##################### run main function #####################

if __name__ == "__main__":
    
    main_function(
                  train_instance_file,
                  train_cluster_file,
                  train_record_file,
                  test_instance_file,
                  test_record_file,
                  blocking_method      = blocking_method,
                  similarity_metric    = similarity_metric,
                  classifier_name_list = classifier_name_list,
                  cluster_file_test    = cluster_file_test,
                  cross_validation     = cross_validation,
                  conduct_classification     = conduct_classification,
                  clustering_algorithm = clustering_algorithm,
                  cluster_blocking_on  = cluster_blocking_on,
                  cluster_count        = cluster_count,
                  threshold_list       = threshold_list,
                  clustering_metric    = clustering_metric,
                  enable_cluster_id    = enable_cluster_id,
                  cluster_id_namespace = cluster_id_namespace
                 )


''' measure finish time '''
elapsed_time_secs = time.time() - start_time
msg = "\nrun time: %s secs" % timedelta(seconds=round(elapsed_time_secs))
print(msg)

### The end of line ###
