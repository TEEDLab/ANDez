### A walk-around of removing an error message about writable path for matplotlib ####
import os    
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib
######################################################################################

import numpy as np
import re
import math
import random
import six
import sys
import unicodedata
import uuid

from unidecode import unidecode
from itertools import groupby
from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from modules import *

##################### define main function #####################

def main_function(
                  train_instance_file,
                  train_cluster_file,
                  train_record_file,
                  test_instance_file,
                  test_record_file,
                  blocking_method      = None,
                  similarity_metric    = None,
                  classifier_name_list = [],
                  cluster_file_test    = None,
                  cross_validation     = 0,
                  conduct_classification     = 1,
                  clustering_algorithm = None,
                  cluster_blocking_on  = 1,
                  cluster_count        = None,
                  threshold_list       = [],
                  clustering_metric    = None,
                  enable_cluster_id    = False,
                  cluster_id_namespace = ""
                 ):
    

    ''' Create instance pairs within blocks for training '''

    train_pairs = get_instance_pairs(
                                     train_instance_file,
                                     train_cluster_file,
                                     blocking_method,
                                     balanced = 0,  # balancing classes > 0 for no; 1 for yes
                                     ratio    = 1,  # sampling instances: 0 < ratio <= 1 (all instances)
                                     verbose  = 1
                                    )

    print( "\n.... Instance pairs for TRAINING created\n" )

           
    ''' Iterate trainig per model '''
    
    for classifier_name in classifier_name_list:  
    
        ''' Train a classifier on training data '''

        disam_model = learn_model(
                                  train_pairs,
                                  train_instance_file,
                                  train_record_file,
                                  classifier_name,
                                  similarity_metric,
                                  cross_validation
                                 )

        print("\n.... A disambiguation model by '" + classifier_name + "' created\n")
        
       
        if conduct_classification:       

            classification_result_list = classification(
                                                        disam_model,
                                                        test_instance_file,
                                                        test_record_file,
                                                        cluster_file_test,
                                                        blocking_method,
                                                        print_result = 1 # <- 0 if no print
                                                       )
            
            if classification_result_list:
                
                file_name = "_".join(['report_classfication_results',
                                       similarity_metric,
                                       classifier_name]) + '.txt'
                
                output_file = open(file_name, 'w')

                for line in classification_result_list:
                    output_file.write(line + "\n")
                
                output_file.close()
                
                print("    '" + file_name + "' created\n")
            
            
        
        ''' produce clustering score and predicted clusters '''
        result_per_block_file_name = "_".join(['report_result_per_block',
                                                similarity_metric,
                                                classifier_name,
                                                clustering_algorithm]) + '.txt'  
            # <- produce a file that contains scores for resuls per block
            # if 'result-per-block' is None, no file is created
        
        print("\n======== Clustering Report ========\n")
        print("Clustering performance is measured by '" + clustering_metric + "'\n")
        
        clustering_result_list = clustering(
                                            test_instance_file,
                                            test_record_file,
                                            disam_model,
                                            cluster_file_test,
                                            n_jobs               = 1, 
                                            clusterer_blocking   = cluster_blocking_on,
                                            n_clusters           = cluster_count,               
                                            clustering_algorithm = clustering_algorithm,
                                            result_per_block     = result_per_block_file_name,
                                            clustering_method    = "average",
                                            threshold_list       = threshold_list,
                                            clustering_metric    = clustering_metric, 
                                            blocking_method      = blocking_method
                                           )
                  
        ''' display clustering evaluation score and print cluster list '''
        
        if len(clustering_result_list) == 1:
            """ only one threshold is provided"""
            
            print("clustering threshold:", clustering_result_list[0][0])
            print("count of blocks:", clustering_result_list[0][1])
            print("count of predicted clusters :", clustering_result_list[0][2])
            print("count of true clusters :", clustering_result_list[0][3])
            #[threshold, count_block, count_pred, count_true, score_list, cluster_list]
            
            ''' print clustering results score '''
            score_list = clustering_result_list[0][4]
            if score_list:
                format_style = "{:.4f}|{:.4f}|{:.4f}"
                print("\nOverall evaluation scores" + "\nprecision|recall|f1-score")
                print(format_style.format(
                                          score_list[0], # <- precision
                                          score_list[1], # <- recall
                                          score_list[2]  # <- f1-score
                                         )
                     )
            
            ''' print clustering results '''
            cluster_list = clustering_result_list[0][5]
            if cluster_list:
                file_name = "_".join(['report_clustering_results',
                                       similarity_metric,
                                       classifier_name,
                                       clustering_algorithm]) + '.txt'
                
                with open(file_name, 'w') as output_file:
                    if enable_cluster_id:
                        # Generate UUID for each cluster in the cluster list
                        for index, line in enumerate(cluster_list):
                            name = classifier_name + '_' + str(index)
                            uuid_obj = uuid.uuid5(cluster_id_namespace, name)
                            output_file.write(str(uuid_obj) + "\t" + line + "\n")                   
                    else:
                        # If enable_cluster_id is False, simply write cluster_list to output_file
                        for line in cluster_list:
                            output_file.write(line + "\n")

                print("\n    'report_clustering_results_" + similarity_metric + '_' + classifier_name + '_' + clustering_algorithm + ".txt' created\n")
        
        else:
            """ a list of threshold is provided """
            print("    Clustering is conducted with various thresholds\n")
            
            threshold_list = []
            precision_list = []
            recall_list = []
            fscore_list = []
            
            for clustering_result in clustering_result_list:
            
                threshold_list.append(round(clustering_result[0],    4))
                precision_list.append(round(clustering_result[4][0], 4))
                recall_list.append(   round(clustering_result[4][1], 4))
                fscore_list.append(   round(clustering_result[4][2], 4))
                
            
            score_line_list = []
            for k in range(len(threshold_list)):
                score_line = "    {}|{}|{}|{}".format(
                                              str(threshold_list[k]),
                                              str(precision_list[k]),
                                              str(recall_list[k]),
                                              str(fscore_list[k])
                                             )
                score_line_list.append(score_line)
            
            header       = "    threshold|precision|recall|f-score"
            print(header + "\n" + "\n".join(score_line_list) + "\n")
            
            ''' display a plot of evaluation scores '''
            print("A figure reporting evaluating scores per threshold is created!\n....Close the figure to end the process\n")
            fig = plt.figure(figsize=(15, 5))
            plt.plot(threshold_list, precision_list, color='blue',  marker='x', linestyle='solid',   linewidth=2, markersize=8, label="Precision")
            plt.plot(threshold_list, recall_list,    color='red',   marker='*', linestyle='dashdot', linewidth=2, markersize=8, label="Recall")
            plt.plot(threshold_list, fscore_list,    color='black', marker='o', linestyle='dashed',  linewidth=2, markersize=8, label="f-score")
            plt.grid(True)
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.title('Clustering Evaluation Score Per Threshold')
            plt.xlabel('Threshold')
            plt.ylabel('Evaluation Score')
            plt.legend()
            plt.show()
            

######################################################################

""" Handeling input data """

IS_PYTHON_3 = sys.version_info[0] == 3

""" Encoding and decoding of input data """

def asciify(string):
    """ Transliterate a string to ASCII """
    if not IS_PYTHON_3 and not isinstance(string, unicode):
        string = unicode(string, "utf8", errors="ignore")

    string = unidecode(unicodedata.normalize("NFKD", string))
    string = string.encode("ascii", "ignore")
    string = string.decode("utf8")

    return string

DROPPED_AFFIXES = {'a', 'ab', 'am', 'ap', 'abu', 'al', 'auf', 'aus', 'bar',
                   'bath', 'bat', 'ben', 'bet', 'bin', 'bint', 'd', 'da',
                   'dall', 'dalla', 'das', 'de', 'degli', 'del', 'dell',
                   'della', 'dem', 'den', 'der', 'di', 'do', 'dos', 'dr', 'ds', 'du',
                   'e', 'el', 'i', 'ibn', 'im', 'jr', 'l', 'la', 'las', 'le',
                   'los', 'm', 'mac', 'mc', 'mhic', 'mic', 'o', 'phd', 'ter', 'und',
                   'v', 'van', 'vom', 'von', 'zu', 'zum', 'zur'}

DROPPED_SUFFICES = {'i', 'ii', 'iii', 'iv', 'v', 'vi', 'jr', 'dr', 'phd'}

def normalize_name(name, drop_affixes=True):
    
    """
    convert a name string into a standard format for processing
    :param name: string as formatted as 'surname, forename'
    :param drop_affixes: boolean
    :return: normalized name

    """

    ### clean string ###
    name = asciify(name).strip().lower() # converted to ASCII, whitespaces deleted, and lowercased
    tokens = name.split(",", 1)

    def _remove_characters(x, y):
        """ replace non-alphabet characters in list x with string y """
        return re.sub("[^a-zA-Z, \s]+", y, x)

    def _remove_tokens(x, y):
        """ remove tokens in dictionary y from list x """
        return list(filter(lambda x: x not in y, x))

    #### decide the surname token of a name without a comma ###
    if len(tokens) == 1: # a string without a comma: e.g., "wei wang" -> "wang, wei"

        token_list = name.split(" ")
        token_list = [_remove_characters(x, "") for x in token_list] # <- no space 
        token_list_filtered = _remove_tokens(token_list, DROPPED_SUFFICES)
        if len(token_list_filtered) > 1: # last token becomes surname
            tokens = [token_list_filtered[-1], " ".join(token_list_filtered[:-1])]
        else:
            tokens = [token_list_filtered[0], ""]

    elif len(tokens) > 1: # a string with a comma

        if drop_affixes:
            surname_list = _remove_characters(tokens[0], " ").split(" ") # <- space
            surname_list_filtered = _remove_tokens(surname_list, DROPPED_AFFIXES)
            if len(surname_list_filtered) > 0:
                tokens[0] = "".join(surname_list_filtered) # <- no space
        else:
            tokens[0] = re.sub('\s', '', tokens[0])

        tokens[1] = _remove_characters(tokens[1], " ")

    else: # empty string
        return ""

    ### glue tokens ### 
    name = " ".join(tokens)

    return name


""" Retrieve information from input data """

def get_author_full_name(X):
    """
    Get author full name from the signature.
    
    :param X: instance dictionary
    :returns: normalized name
    
    """
    name = X['author_name']
    name = normalize_name(name) if name else ""

    return name

def get_coauthor(X, range=10, simplified=1):

    """
    Get a string of concatenated coauthor names

    :param X: instance dictionaries
    :param range: integer
        coauthor names within integer away from author are filtered
        implemented due to hyper-authorship -> computational burden
    :param initialized: yes = 1, no = otherwise
        simplify coauthor names into surname and first forename initial
        increase recall and computational efficiency
    
    :returns: string of coauthor names separated by space
    
    """
    author_list = X['paper']['authors'] # <- list []
    

    try:
        ### find index of author name among author_list ###
        ''' get author name '''
        author_name = X['author_name']
        ''' get index of author name in authors list '''
        index = author_list.index(author_name)
        ''' filter coauthor names '''
        if len(author_list) <= range:  # <- most papers have less than 10 authors
            author_list_filtered = author_list
        else:
            author_list_filtered = author_list[max(0, index-range) : min(len(author_list), index+range) + 1]
                # <- don't forget 'inclusive -> [ : ] <-exclusive'
        ''' remove author name from filtered names '''
        author_list_filtered.remove(author_name)
        ''' convert coauthor name into a 'full surname and first forename initial' form '''
        if simplified == 1:
            author_list_str = " ".join([get_first_initial(x) for x in author_list_filtered]) 
        else:
            author_list_str = " ".join(author_list_filtered)
            
        return author_list_str

    except ValueError:
        
        if simplified == 1:
            author_list_str = " ".join([get_first_initial(x) for x in author_list]) 
        else:
            author_list_str = " ".join(author_list)

        return author_list_str

def clean_string_1(string):
    
    """
    Clean input string by removing non alphabetical characters
    """
    
    string = re.sub("[^a-zA-Z\s]+", " ", asciify(string).lower())
    string = re.sub("\s+", " ", string).strip()

    return string

def clean_string_2(string):
    
    """
    Clean input string by removing non alpha-numeric characters
    """
    
    string = re.sub("[^0-9a-zA-Z\s]+", " ", asciify(string).lower())
    string = re.sub("\s+", " ", string).strip()

    return string

def get_affiliation(input_dic):
    
    """
    Get affiliation string associated with an author
    :param input_dic: instance dictionary
    :returns: cleaned string 
    
    """
    
    string = input_dic['affiliation']
    string = clean_string_1(string) if string else ""
    
    return string

def get_email_address(input_dic):
    
    """
    Get emaial address string associated with an author
    :param input_dic: instance dictionary
    :returns: cleaned string 
    
    """
    
    string = input_dic['email_address']
    string = clean_string_2(string) if string else ""
    
    return string

def get_title(input_dic):
    
    """
    Get title string associated with an author's paper
    :param input_dic: instance dictionary
    :returns: cleaned string 
    
    """
    
    string = input_dic['paper']['title']
    string = clean_string_1(string) if string else ""
    
    return string
    
def get_venue(input_dic):
    
    """
    Get venue (journal or conference) string associated with an author's paper
    :param input_dic: instance dictionary
    :returns: cleaned string 
    
    """
    
    string = input_dic['paper']['venue']
    string = clean_string_1(string) if string else ""
            # may not conduct cleaning if input data are cleaned
    
    return string
    
def get_keyword(input_dic):
    
    """
    Get keyword string associated with an author's paper
    :param input_dic: instance dictionary
    :returns: cleaned string 
    
    """
    
    list = input_dic['paper']['keyword']
    string = " ".join(list)
    
    return string

def group_by_instance(r):
    
    """
    Grouping function for PairTransformer
    :param r: iterable instance in a singleton
    :returns: string of instance id
    
    """
    return r[0]["instance_id"]


""" Blocking methods """

def get_first_initial(name):

    """ convert name string into surname and first forename initial """
    
    tokens = normalize_name(name).split(" ", 1)

    try:
        name = "%s %s" % (tokens[0], tokens[1].strip()[0])
    except IndexError:
        name = tokens[0]

    return name

def block_by_first_initial(X):

    """
    Convert names into a format of surname followed by first forename initial
    and generate np.array that contains them

    :param X: numpy array of instance dictionaries
    :returns: numpy array of block names
            e.g., 'Kim, Jinseok' -> 'kim j'
            The order of the array is the same as in X
    
    """

    blocks = []

    for instance in X[:, 0]:
        blocks.append(get_first_initial(instance['author_name']))

    ### don't forget return np.array ###
    return np.array(blocks)

def block_by_full_name(X):
    
    """
    Use normailized names as block keys and generate np.array that contains them

    :param X: numpy array of instance dictionaries
    :returns: numpy array of block names
            e.g., 'Kim, Jinseok' -> 'kim jinseok'
            The order of the array is the same as in X
    
    """
    
    def _get_full_name(name):
        
        tokens = normalize_name(name).split(" ", 1)

        try:
            tokens[1] = tokens[1].replace(" ", "") # remove spaces in forename
            name = "%s %s" % (tokens[0], tokens[1].strip())
        except IndexError:
            name = tokens[0]

        return name

    blocks = []

    for instance in X[:, 0]:
        blocks.append(_get_full_name(instance['author_name']))

    ### don't forget return np.array ###
    return np.array(blocks)

def block_by_forename_strip(X):

    """
    Use names wiht forename stripped as block keys
    and generate np.array that contains them

    :param X: numpy array of instance dictionaries
    :returns: numpy array of block names
            e.g., 'Kim, Jinseok' -> 'kim jin' (n = 3)
            The order of the array is the same as in X
    
    """

    def _strip_forename(name):

        """ keep first n characters of a name string """
        tokens = normalize_name(name).split(" ", 1)

        try:
            tokens[1] = tokens[1].replace(" ", "")
            name = "%s %s" % (tokens[0], tokens[1].strip()[:3]) # <- modify n here
        except IndexError:
            name = tokens[0]

        return name

    blocks = []

    for instance in X[:, 0]:
        blocks.append(_strip_forename(instance['author_name']))

    return np.array(blocks)


""" Convert input files """

def get_cluster_dic(cluster_file):
    
    """
    convert cluster file into cluster dictionary 
    :param cluster_file: text file of cluster ids and instance ids
    :returns: dict{unicode:list[int], unicode:list[int],...}
    
    """

    cluster_dic = {}
    for line in cluster_file:
        
        info_list = line.split('\t')
        cluster_id = info_list[0]   
        cluster_dic[cluster_id] = [int(i) for i in info_list[1].strip().split('|')]
    
    return cluster_dic

def get_instance_dic_list(instance_file):
    
    """
    convert instance file into instance list
    :param instance_file: text file of instance information
    :returns : list of dictionaries

    """
    
    instance_dic_list = []
    
    for line in instance_file:
        
        # get information from line
        info_list = line.split('\t')

        # create a dictionary
        instance_dic = {}
        
        instance_dic['instance_id']   = int(info_list[0])
        instance_dic['paper_id']      = int(info_list[1])
        instance_dic['name_position'] = int(info_list[2])
        instance_dic['author_name']   =     info_list[3]
        #instance_dic['affiliation']   = info_list[X].strip()
        #instance_dic['email_address'] = info_list[Y].strip()
        
        # append dic to list
        instance_dic_list.append(instance_dic)
 
    return instance_dic_list

def get_record_dic_list(record_file):
    """
    convert record file into record list
    :param record_file: text file of record information
    :returns: list of dictionaries

    """

    record_dic_list = []
    
    for line in record_file:
        
        # get information from line
        info_list = line.split('\t')

        # create a dictionary    
        record_dic = {}
        record_dic['paper_id'] =     int(info_list[0])
        record_dic['year']     = None if info_list[1] == '' else int(info_list[1])
        record_dic['venue']    = None if info_list[2] == '' else info_list[2]
        record_dic['authors']  = []   if info_list[3] == '' else [author for author in info_list[3].split('|')]
        record_dic['title']    = None if info_list[4] == '' else info_list[4].strip()
        #record_dic['keyword'] = []   if info_list[X] == '' else [keyword for keyword in info_list[X].split('|')]
        
        # append dic to list
        record_dic_list.append(record_dic)

    return record_dic_list

def get_instance_dic_dic(instance_file, record_file):
    """
    convert instance file and record file into a dictionary of instance dictionaries
    :param instance_file: text file of instance information
    :param record_file: text file of record information
    :returns: instance_dic_dic
    
    """
    
    with open(instance_file, 'r', encoding="utf8") as instancefile:          
        instance_dic_list = get_instance_dic_list(instancefile)
    
    with open(record_file,   'r', encoding="utf8") as recordfile:        
        record_dic_list   = get_record_dic_list(recordfile)
   
    if isinstance(instance_dic_list, list):
        instance_dic_dic = {instance_dic['instance_id']: instance_dic for instance_dic in instance_dic_list}

    if isinstance(record_dic_list,   list):
        record_dic_dic   = {record_dic['paper_id']     : record_dic   for record_dic   in record_dic_list}

    for idx, instance_dic in instance_dic_dic.items():
        instance_dic['paper'] = record_dic_dic[instance_dic['paper_id']]
    
    return instance_dic_dic


""" Generate instance pairs within blocks """

def get_instance_pairs(
                  instance_file,
                  cluster_file,
                  blocking_method,
                  balanced = 0,
                  ratio = 1,
                  verbose = 0
                  ):
        
    """
    generate a list of instance pairs to be compared for similarity
    :param instance_file: text file of instance information
    :param cluster_file: text file of cluster information
            -> if None, return list of tuples without match (0) or nonmatch (1) labels
    :param blocking_method: string of blockign method 
    :param balanced: if yes, 1; otherwise, 0
        for more details on training data imbalance, see a paper below
        Kim, J., & Kim, J. (2018). The impact of imbalanced training data on machine learning
                    for author name disambiguation. Scientometrics, 117(1), 511-526. 
    
    :param ratio: float number between zero and one
    :param verbose: if yes, 1; otherwise, 0
    :returns: list of tuples that contain instance ids of instance pairs and label

    """
    
    
    ''' load instance file '''
    with open(instance_file, 'r', encoding = "utf8") as instancefile:
        instance_dic_list = get_instance_dic_list(instancefile)
        

    input_list = []
    for instance_dic in instance_dic_list:
        input_list.append([instance_dic])

    input_list_array = np.array(input_list)

    if   blocking_method == "first_initial":
        block_array = block_by_first_initial(input_list_array)
    elif blocking_method == "full_name":
        block_array = block_by_full_name(input_list_array)
    elif blocking_method == "forename_strip":
        block_array = block_by_forename_strip(input_list_array)
        
    ''' get dictionary of block name and its indices '''
    block_dic = {}

    for index, block_name in enumerate(block_array):
        if block_name in block_dic:
            block_dic[block_name].append(index)
        else:
            block_dic[block_name] = [index]

    all_pairs = []
    
    if cluster_file: # <- cluster file provided for training or evaluation
    
        ''' load cluster file '''
        with open(cluster_file, 'r') as clusterfile:
            cluster_dic = get_cluster_dic(clusterfile)
        
        ''' create a dictionary of instance id and cluster id '''
        clusters_reversed = {v: k for k, va in six.iteritems(cluster_dic) for v in va}
        
        ''' get positive and negative pairs '''
        positive_pairs = []
        negative_pairs = []
        
        for _, index_list in block_dic.items():
            ''' compare all instances pairwisely within blocks for labels '''
            for i, index_1 in enumerate(index_list):
                for index_2 in index_list[i+1:]:
                    
                    instance_id_1 = instance_dic_list[index_1]['instance_id']
                    instance_id_2 = instance_dic_list[index_2]['instance_id']
                    cluster_id_1  = clusters_reversed[instance_id_1]
                    cluster_id_2  = clusters_reversed[instance_id_2]
                    
                    if cluster_id_1 == cluster_id_2: # Same author
                        positive_pairs.append((instance_id_1, instance_id_2, 0))
                    else: # Different authors
                        negative_pairs.append((instance_id_1, instance_id_2, 1))

        ''' aggregate positive and negative pairs '''
        all_pairs = positive_pairs + negative_pairs
        
        all_len = len(all_pairs)
        pos_len = len(positive_pairs)
        neg_len = len(negative_pairs)
       
        print("Instance pairs created...\n")
        
        if verbose:
            print("Count of Pairs")
            print("    all|positive|negative : " + str(all_len) + "|" + str(pos_len) + "|" + str(neg_len) + "\n")
        
        ''' create pair subset in which both positive and negative pairs are equal in size '''
        if balanced:
            print("Balanced sampling (positive:negative = 1:1) conducted...")
            
            if pos_len == neg_len and pos_len > 0:
                print("    positive and negative pairs are same in length")
            elif pos_len > neg_len and neg_len > 0:
                all_pairs = negative_pairs + random.sample(positive_pairs, neg_len)
                print("    positive > negative: balanced total (" + str(len(all_pairs)) + ")\n")
            elif pos_len < neg_len and pos_len > 0:
                all_pairs = positive_pairs + random.sample(negative_pairs, pos_len) 
                print("    positive < negative: balanced total (" + str(len(all_pairs)) + ")\n")
            else:
                print ("!Warning! Check count of pairs...\n")
        else:
            ''' issue warning if one or both are zero in length '''
            if (pos_len == 0) or (neg_len == 0):
                all_pairs = []
                print("!Warning! Positive or negative pairs length is 0\n")

        ''' reduce size of pairs '''
        if ratio == 1:
            print ("Ratio sampling NOT conducted...\n")
            
        elif ratio < 1 and ratio > 0:
            all_pairs = random.sample(all_pairs, int(len(all_pairs)*ratio))
            print("Ratio (" + str(ratio) + ") sampling done...")
            print("    sample size: " + str(len(all_pairs)))
        else:
            print ("!Warning! Check sampling ratio...\n")
            
    else: # <- cluster file is unavailable
        
        for _, index_list in block_dic.items():
            ''' compare all instances pairwisely within blocks for labels '''
            for i, index_1 in enumerate(index_list):
                for index_2 in index_list[i+1:]:
                    
                    instance_id_1 = instance_dic_list[index_1]['instance_id']
                    instance_id_2 = instance_dic_list[index_2]['instance_id']
                    all_pairs.append((instance_id_1, instance_id_2, ''))

        print("A total of ", len(all_pairs), " instance pairs (without labels) are created\n")

    return all_pairs


""" Model training and evaluation """

def build_prediction_model(
                              X, y, 
                              classifier_name, 
                              similarity_metric,
                              cross_validation  = 0,
                              ):
    """
    Build a vector reprensation of instance pairs.
    
    :param X: an instance pair
    :returns: estimator 
    
    """ 
    
    ''' build transformer for similarity calculation '''
    # URL ->  https://scikit-learn.org/stable/modules/compose.html
    if similarity_metric == "cos" or similarity_metric == "jac":
        
        pair_author_name = PairTransformer(element_transformer=Pipeline([
            ("full_name", FuncTransformer(func=get_author_full_name)),
            ("shaper", Shaper(newshape=(-1,))),
            ("tf-idf", TfidfVectorizer(analyzer="char_wb",
                                       ngram_range=(2, 4),
                                       dtype=np.float32,
                                       use_idf = False,
                                       decode_error="replace")),
            ]), groupby=group_by_instance)
        
        pair_coauthor = PairTransformer(element_transformer=Pipeline([
            ("coauthor", FuncTransformer(func=get_coauthor)),
            ("shaper", Shaper(newshape=(-1,))),
            ("tf-idf", TfidfVectorizer(analyzer="char_wb",
                                       ngram_range=(2, 4),
                                       dtype=np.float32,
                                       use_idf = False,
                                       decode_error="replace")),
            ]), groupby=group_by_instance)
        
        #pair_affiliation = PairTransformer(element_transformer=Pipeline([
        #    ("affiliation", FuncTransformer(func=get_affiliation)),
        #    ("shaper", Shaper(newshape=(-1,))),
        #    ("tf-idf", TfidfVectorizer(analyzer="char_wb",
        #                               ngram_range=(2, 4),
        #                               dtype=np.float32,
        #                               decode_error="replace")),
        #    ]), groupby=group_by_instance)
        
        #pair_email_address = PairTransformer(element_transformer=Pipeline([
        #    ("email", FuncTransformer(func=get_email_address)),
        #    ("shaper", Shaper(newshape=(-1,))),
        #    ("tf-idf", TfidfVectorizer(analyzer="char_wb",
        #                               ngram_range=(2, 4),
        #                               dtype=np.float32,
        #                               decode_error="replace")),
        #    ]), groupby=group_by_instance)

        pair_title = PairTransformer(element_transformer=Pipeline([
            ("title", FuncTransformer(func=get_title)),
            ("shaper", Shaper(newshape=(-1,))),
            ("tf-idf", TfidfVectorizer(analyzer="char_wb",
                                       ngram_range=(2, 4),
                                       dtype=np.float32,
                                       use_idf = False,
                                       decode_error="replace")),
            ]), groupby=group_by_instance)
        
        pair_venue = PairTransformer(element_transformer=Pipeline([
            ("venue", FuncTransformer(func=get_venue)),
            ("shaper", Shaper(newshape=(-1,))),
            ("tf-idf", TfidfVectorizer(analyzer="char_wb",
                                       ngram_range=(2, 4),
                                       dtype=np.float32,
                                       decode_error="replace")),
            ]), groupby=group_by_instance)           
        
        #pair_keyword = PairTransformer(element_transformer=Pipeline([
        #    ("keyword", FuncTransformer(func=get_keyword)),
        #    ("shaper", Shaper(newshape=(-1,))),
        #    ("tf-idf", TfidfVectorizer(analyzer="char_wb",
        #                               ngram_range=(2, 4),
        #                               dtype=np.float32,
        #                               use_idf = False,
        #                               decode_error="replace")),
        #    ]), groupby=group_by_instance)

        if similarity_metric   == "cos":
            combiner = CosineSimilarity()
        elif similarity_metric == "jac":
            combiner = JaccardSimilarity()
    else:        
        pair_author_name = PairTransformer(element_transformer=FuncTransformer(
            func=get_author_full_name), groupby=group_by_instance)
        
        pair_coauthor = PairTransformer(element_transformer=FuncTransformer(
            func=get_coauthor), groupby=group_by_instance)
        
        #pair_affiliation = PairTransformer(element_transformer=FuncTransformer(
        #    func=get_affiliation), groupby=group_by_instance)
        
        #pair_email_address = PairTransformer(element_transformer=FuncTransformer(
        #    func=get_email_address), groupby=group_by_instance)       

        pair_title = PairTransformer(element_transformer=FuncTransformer(
            func=get_title), groupby=group_by_instance)
        
        pair_venue = PairTransformer(element_transformer=FuncTransformer(
            func=get_venue), groupby=group_by_instance)

        #pair_keyword = PairTransformer(element_transformer=FuncTransformer(
        #    func=get_keyword), groupby=group_by_instance)
        
        if similarity_metric == 'jrw':
            combiner = StringDistance(similarity_function = 'jaroWin')     
    
    ''' concatenate results of multiple transformer objects '''
    #https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html
    transformer = FeatureUnion([
                                ('author_name_similarity', Pipeline([
                                    ('pairs', pair_author_name),
                                    ('combiner', combiner)
                                ])),
                                ('coauthor_similarity', Pipeline([
                                    ('pairs', pair_coauthor),
                                    ('combiner', combiner)
                                ])),
                                #('affiliation_similarity', Pipeline([
                                #    ('pairs', pair_affiliation),
                                #    ('combiner', combiner)
                                #])),
                                #('email_address_similarity', Pipeline([
                                #    ('pairs', pair_email_address),
                                #    ('combiner', combiner)
                                #])),
                                ('title_similarity', Pipeline([
                                    ('pairs', pair_title),
                                    ('combiner', combiner)
                                ])),
                                ('venue_similarity', Pipeline([
                                    ('pairs', pair_venue),
                                    ('combiner', combiner)
                                #])), # <<< be careful to commentize this
                                #('keyword_similarity', Pipeline([
                                #    ('pairs', pair_keyword),
                                #    ('combiner', combiner)
                                ])) # <<< be careful not to commentize this         
                               ])

    ''' select classifiers '''
 
    if classifier_name == "RF":
        ''' Random Forest ''' 
        # URL ->  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        classifier = RandomForestClassifier(
                                n_estimators = 500,
                                n_jobs       = 8,
                                random_state = 1,
                                verbose      = 0
                                )
    elif classifier_name == "GB":
        ''' Gradient Boosting '''
        # URL ->  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        classifier = GradientBoostingClassifier(
                                    n_estimators  = 500,
                                    max_depth     = 9,
                                    max_features  = 5, # <- !CAUTION! check max number of features
                                    learning_rate = 0.125,
                                    verbose       = 0
                                    )
    elif classifier_name == "NB":
        ''' Naive Bayes '''
        # URL ->  https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
        classifier = GaussianNB()
    elif classifier_name == "LR":
        ''' Logistic Regression '''
        # URL ->  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        classifier = LogisticRegression(random_state=1) 
    elif classifier_name == "SVM":
        ''' Support Vector Machine '''
        # URL ->  https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        classifier = SVC(kernel='linear', probability=True)
    elif classifier_name == "DT":
        ''' Decision Tree '''
        # URL ->  https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        classifier = DecisionTreeClassifier(random_state=0)   
    else:
        raise ValueError("Must be a valid classifier name!")
        
    
    ''' combine transformer and classifier to have a final estimator '''
    # URL ->  https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    pipe = Pipeline([("transformer", transformer),("classifier", classifier)])

    ''' fit the pipeline on the whole training data '''
    estimator = pipe.fit(X, y)
    
    ''' conduct 10-fold cross validation, if selected '''
    if cross_validation:

        from sklearn.model_selection import StratifiedKFold
        # URL ->  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

        skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
        iter_num = 1

        class0_pre_list = [] # <- precision
        class0_rec_list = [] # <- recall
        class0_fsc_list = [] # <- f1-score
        class1_pre_list = []
        class1_rec_list = []
        class1_fsc_list = []
        rmse_list  = []
        
        print("10-fold cross validation is being conducted...")
        
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = pipe.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            pre, rec, fsc, support = metrics.precision_recall_fscore_support(y_test, y_pred)
            rmse   = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
            matrix = metrics.confusion_matrix(y_test, y_pred)
            acc    = metrics.accuracy_score(y_test, y_pred)

            print("\nRound-" + str(iter_num) + ": Precision|Recall|F1|Support")
            print("-class0: " + "{:.4f}|{:.4f}|{:.4f}|{}".format(pre[0], rec[0], fsc[0], support[0]))
            class0_pre_list.append(pre[0])
            class0_rec_list.append(rec[0])
            class0_fsc_list.append(fsc[0])
            
            print("-class1: " + "{:.4f}|{:.4f}|{:.4f}|{}".format(pre[1], rec[1], fsc[1], support[1]))
            class1_pre_list.append(pre[1])
            class1_rec_list.append(rec[1])
            class1_fsc_list.append(fsc[1])
            
            print("-RMSE: " + "{:.4f}".format(rmse))
            rmse_list.append(rmse)
            print("-Confusion Matrix:")
            print(matrix)

            iter_num += 1

        ''' compute mean and standard deviation of entire scores for each class '''
        class0_pre_array = np.array(class0_pre_list)
        class0_rec_array = np.array(class0_rec_list)
        class0_fsc_array = np.array(class0_fsc_list)

        class1_pre_array = np.array(class1_pre_list)
        class1_rec_array = np.array(class1_rec_list)
        class1_fsc_array = np.array(class1_fsc_list)

        rmses = np.array(rmse_list)
        
        print("Mean(SD): precision|recall|f1")
        format_style = "{:.4f}({:.4f})|{:.4f}({:.4f})|{:.4f}({:.4f})"
        print("-class0: " + format_style.format(
                                          class0_pre_array.mean(), class0_pre_array.std(),
                                          class0_rec_array.mean(), class0_rec_array.std(),
                                          class0_fsc_array.mean(), class0_fsc_array.std()
                                         )
             )
        
        print("-class1: " + format_style.format(
                                          class1_pre_array.mean(), class1_pre_array.std(),
                                          class1_rec_array.mean(), class1_rec_array.std(),
                                          class1_fsc_array.mean(), class1_fsc_array.std()
                                         )
             )

        print("-RMSE: " + "{:.4f}({:.4f})".format(rmses.mean(), rmses.std()) + "\n")

    ''' compute feature importances if random forest is selected '''
    if classifier_name == 'RF':
        
        ''' build a list of feature names ''' 
        feature_label_trans = [transformer.transformer_list[i][0] for i in range(len(transformer.transformer_list))]
        feature_list = []
        for l in range(len(feature_label_trans)):
            feature_list.append(feature_label_trans[l])

        feature_impo = classifier.feature_importances_        
        
        #print('feature count: ' + str(classifier.n_features_) + '\n')
        #print('feature list : ' + str(feature_list) + '\n')
        #print('feature importances: ' + str(feature_impo) + '\n')

        ''' rearrange feature importane in descending order '''
        indices = np.argsort(feature_impo)[::-1]
        print("RF Feature Importance Ranking\n")
        
        for feature_index in range(len(feature_impo)):
            print("%2d) %-*s %f" % (
                                    feature_index + 1, 30,
                                    feature_list[indices[feature_index]],
                                    feature_impo[indices[feature_index]]
                                   )
                 )
            
    return estimator

def learn_model(
                instance_pairs, 
                instance_file, 
                record_file, 
                classifier_name, 
                similarity_metric,
                cross_validation  = 0,
               ):
    
    """
    Learn a prediction model (for classification) or distance estimator (for clustering)
    :param pairs: training instance pairs that contain match or non-match labels
    :param instance_file: text file of instance information
    :param record_file: text file of record information
    :param classifier_name:
    :param similarity metric:
    :param cross_validation:
    :return: disam model (= distance estimator)
    
    """    
    
    ''' create dictionary of instance dictionaries '''
    instance_dic_dic = get_instance_dic_dic(instance_file, record_file)

    ''' create empty numpy objects for instance pairs and labels '''
    instance_pairs_len = len(instance_pairs)
    X = np.empty((instance_pairs_len, 2), dtype=np.object)
    y = np.empty(instance_pairs_len, dtype=np.int)

    ''' create input for training classifiers '''
    for k, (i, j, target) in enumerate(instance_pairs):
        X[k, 0] = instance_dic_dic[i]
        X[k, 1] = instance_dic_dic[j]
        y[k]    = target # <- match (0) or non-match (1) label
    
    ''' produce prediction model (distance estimator) '''
    disam_model = build_prediction_model(
                                              X, y, 
                                              classifier_name, 
                                              similarity_metric,
                                              cross_validation  = cross_validation
                                              )

    return disam_model

def classification(
                   disam_model,  
                   instance_file,
                   record_file,
                   cluster_file,
                   blocking_method,
                   print_result = 0
                  ):
    
    """
    
    
    """

    ''' generate instance pairs for prediction of test data '''
    instance_pairs = get_instance_pairs(
                                        instance_file,
                                        cluster_file,
                                        blocking_method,
                                        balanced = 0, # <- no balancing
                                        ratio    = 1, # <- no sampling
                                        verbose  = 1
                                       )
    
    print(".... Instance pairs for TEST created\n")
    
    ''' create empty numpy objects for instance pairs and labels '''
    instance_pairs_len = len(instance_pairs)
    Xt = np.empty((instance_pairs_len, 2), dtype=np.object)
    yt = np.empty(instance_pairs_len, dtype=np.int)
    
    ''' create dictionary of instance dictionaries '''
    instance_dic_dic = get_instance_dic_dic(instance_file, record_file)

    ''' create input for prediction '''
    for k, (i, j, target) in enumerate(instance_pairs):
        Xt[k, 0] = instance_dic_dic[i]
        Xt[k, 1] = instance_dic_dic[j]
        yt[k]    = target if cluster_file else -9 # -9 if 'cluster_file' unavailable
    
    ''' acess transformer object from the pipeline in learn_model() '''
    trans_est = disam_model.named_steps['transformer']
    Xt_trans  = trans_est.transform(Xt)  
    
    ''' get prediction probability and predicted class (= target = label) '''
    yt_prob       = disam_model.predict_proba(Xt)[:, 1]
    yt_pred_class = disam_model.predict(Xt)
    
    ''' display confusion matrix and evaluation score, if test cluster is available '''
    print('======== Classification Report ========\n')
    if cluster_file is not None:
    
        ''' print confusion matrix '''   
        confusion = metrics.confusion_matrix(yt, yt_pred_class)
        print('Confusion Matrix:\n')
        print(confusion)
        
        ''' print evaluation scores '''
        print('\nEvaluation Score:\n')
        print(metrics.classification_report(yt, yt_pred_class, target_names = ['class 0', 'class 1']))       
    
    ''' generate a list of classification results '''
    classification_result_list = []
    if print_result:
        
        ''' extract feature names '''
        feature_list = [trans_est.transformer_list[i][0].split('_')[0] for i in range(len(trans_est.transformer_list))]
        feature_label = '|'.join(feature_list)

        ''' create a header for output file '''
        header = "pair_id|inst_id_1|inst_id_2|" + feature_label + "|class_prob|predicted_class|true_class"
        classification_result_list.append(header)

        for i in range(instance_pairs_len):
            try:
                instance_id_1 = Xt[i, 0]['instance_id']
                instance_id_2 = Xt[i, 1]['instance_id']
                
                sim_score_list= [str(round(Xt_trans[i][j], 4)) for j in range(len(feature_list))]                
                sim_scores = '|'.join(sim_score_list)

                y_prob = round(yt_prob[i], 4)
                y_pred = yt_pred_class[i]
                y_true = yt[i]

                output_line = "|".join(
                                       [str(i+1),
                                        str(instance_id_1), str(instance_id_2),
                                        sim_scores,
                                        str(y_prob),
                                        str(y_pred),
                                        str(y_true)]
                                      )
                
                classification_result_list.append(output_line)
            
            except IndexError:
                print("IndexError: " + str(i))
                continue

    return classification_result_list
    
    
""" Clustering """

def _affinity(X, step=10000, clustering_algorithm=None):
    
    """ 
    Create affinity function using a distance estimator learned from training
    :param X:
    :param step:
    :param clustering_algorithm:
    :return distances:
    
    """
    
    ''' create distances between instances using distance_estimtor'''
    all_i, all_j = np.triu_indices(len(X), k=1)
    n_pairs   = len(all_i)
    distances = np.zeros(n_pairs, dtype=np.float64)

    for start in range(0, n_pairs, step):
        end = min(n_pairs, start+step)
        Xt  = np.empty((end-start, 2), dtype=np.object)

        for k, (i, j) in enumerate(zip(all_i[start:end], all_j[start:end])):
            Xt[k, 0], Xt[k, 1] = X[i, 0], X[j, 0]
        
        Xt = distance_estimator.predict_proba(Xt)[:, 1] # <- global distance_estimator
        distances[start:end] = Xt[:]
        
    ''' convert distances into square form '''
    clustering_algorithm_list = ["db", "kmeans", "agg", "spectral"] 
        # <- if other clustering algorithms are added, don't forget adding its name here

    if clustering_algorithm in clustering_algorithm_list:
        from scipy.spatial.distance import squareform
        distances = squareform(distances)
            
    return distances

def clustering(
               instance_file,
               record_file,
               disam_model,
               cluster_file_test,
               output_clusters      = None,
               n_jobs               = -1,
               clusterer_blocking   = 1,
               n_clusters           = None,
               clustering_algorithm = None,
               result_per_block     = None,
               clustering_method    = "average",
               threshold_list       = [],
               clustering_metric    = "b-cubed",
               blocking_method      = None
              ):

    # Assumes that 'distance_estimator' lives in global, making things fast
    global distance_estimator
    distance_estimator = disam_model

    try:
        distance_estimator.steps[-1][1].set_params(n_jobs=1)
    except:
        pass

    instance_dic_dic = get_instance_dic_dic(instance_file, record_file)

    indices = {}
    X = np.empty((len(instance_dic_dic), 1), dtype=np.object)
    for i, signature in enumerate(sorted(instance_dic_dic.values(), key=lambda s: s["instance_id"])):
        X[i, 0] = signature
        indices[signature["instance_id"]] = i
    
    ''' assign blocking method '''
    if   blocking_method == "first_initial":
        blocking_function = block_by_first_initial
    elif blocking_method == "full_name":
        blocking_function = block_by_full_name
    elif blocking_method == "forename_strip":
        blocking_function = block_by_forename_strip   

    ''' create numpy object of truth for evaluation, if cluster file is available '''
    y = None
    if cluster_file_test:
        
        with open(cluster_file_test, 'r') as clusterfile:
            true_clusters = get_cluster_dic(clusterfile)

        y = -np.ones(len(X), dtype=np.int) #

        for label, instance_ids in true_clusters.items():
            for instance_id in instance_ids:
                y[indices[instance_id]] = label #

        #y = y_true
    
    ''' assign scoring metric '''
    if clustering_metric == "cluster-f":
        supervised_scoring = clusterf_precision_recall_fscore
    elif clustering_metric == "k-metric":
        supervised_scoring = kmetric_precision_recall_fscore
    elif clustering_metric == "split-lump":
        supervised_scoring = split_lump_error_precision_recall_fscore
    elif clustering_metric == "pairwise-f":
        supervised_scoring = pairwisef_precision_recall_fscore
    elif clustering_metric == "b-cubed":
        supervised_scoring = bcubed_precision_recall_fscore
	
    
    clusterer_list = []
    ''' decide clustering procedure '''  
    if clusterer_blocking:
        
        ''' create a list of thresholds thare evenly spaced over a specific interval '''
        if len(threshold_list) > 1:
            threshold_list = np.linspace(threshold_list[0], # <- starting value
                                         threshold_list[1], # <- end value
                                         threshold_list[2]  # <- number of samples to generate
                                        ).tolist() # <- convert numpy array to list
         
        for threshold in threshold_list:
            
            #print("\nClustering threshold:", threshold, "\n")
        
            clusterer = BlockClustering(
                            blocking         = blocking_function,
                            base_estimator   = ClusteringMethods(
                                                    affinity             = _affinity,
                                                    n_clusters           = n_clusters,
                                                    clustering_algorithm = clustering_algorithm,
                                                    threshold            = threshold,
                                                    method               = clustering_method,
                                                    supervised_scoring   = supervised_scoring
                                                    ),
                            verbose          = 0, # 1: print results per block on screen
                            result_per_block = result_per_block,
                            n_jobs           = n_jobs
                            ).fit(X, y)
                            
            clusterer_list.append(clusterer)
        
    else:
        clusterer = ClusteringWithoutBlocking(
                        affinity             = _affinity,
                        n_clusters           = n_clusters,
                        clustering_algorithm = clustering_algorithm,
                        method               = clustering_method,
                        result_per_block     = result_per_block,
                        supervised_scoring   = supervised_scoring
                        ).fit(X, y)
                        
        clusterer_list.append(clusterer)
    
    
    clustering_result_list = []
    for i in range(len(clusterer_list)):
        
        clusterer = clusterer_list[i]
        threshold = threshold_list[i]
        
        cluster_list = []
        labels = clusterer.labels_
        
        ''' produce a cluster list if (1) a threshold or
            no threshold (= 0) is provided '''
        if len(threshold_list) < 2:
            ''' extract labels '''
 
            ## average threshold score
            #avg_thres = clusterer.avg_threshold_
            
            ''' generate a list of clusters '''
            cluster_dic = {}
            for label in np.unique(labels):
                mask = (labels == label)
                cluster_dic[str(label)] = [r[0]["instance_id"] for r in X[mask]]

            keys   = cluster_dic.keys()
            values = cluster_dic.values()
          
            for value in values:
                instance_ids = '|'.join(map(str, value))
                cluster_list.append(instance_ids)

        ''' produce a list of clustering evaluation scores  '''
        score_list = []
        count_block = 0
        if clusterer_blocking:
            count_block = len(clusterer.clusterers_)
        count_pred = len(np.unique(labels))
        count_true = 0
        
        ''' produce performance score list, if test cluster file is available '''
        if y is not None:          # <- test cluster file is available 
            count_true = len(np.unique(y))
            if clustering_metric == "cluster-f":
                score_list = clusterf_precision_recall_fscore(y, labels)
            elif clustering_metric == "k-metric":
                score_list = kmetric_precision_recall_fscore(y, labels)
            elif clustering_metric == "split-lump":
                score_list = split_lump_error_precision_recall_fscore(y, labels)
            elif clustering_metric == "pairwise-f":
                score_list = pairwisef_precision_recall_fscore(y, labels)
            elif clustering_metric == "b-cubed":
                score_list = bcubed_precision_recall_fscore(y, labels)
				
           
        clustering_result_list.append([threshold, count_block, count_pred, count_true, score_list, cluster_list])
        
    return clustering_result_list
    
### The end of line ###
