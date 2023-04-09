"""
This code contains a substantially modified version of the code from the BEARD library 
The original code is provided by the code authors: Gilles Louppe & Hussein Al-Natsheh
For details on the original code, see a paper below:
Louppe, G., Al-Natsheh, H. T., Susik, M., & Maguire, E. J. (2016). 
Ethnicity Sensitive Author Disambiguation Using Semi-supervised Learning. 
Knowledge Engineering and Semantic Web, Kesw 2016, 649, 272-287. doi:10.1007/978-3-319-45880-9_21

"""

import time
import math 
import numpy as np
import scipy.sparse as sp
import jellyfish
import scipy.cluster.hierarchy as hac

from operator import mul
from itertools import groupby
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import ClusterMixin
from sklearn.utils import column_or_1d
from sklearn.preprocessing import binarize
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering


""" Transformers for paired data and similarity algorithms """

class FuncTransformer(BaseEstimator, TransformerMixin):
    """Apply a given function element-wise."""

    def __init__(self, func, dtype=None):
        """Initialize.

        Parameters
        ----------
        :param func: callable
            The function to apply on each element.

        :param dtype: numpy dtype
            The type of the values returned by `func`.
            If None, then use X.dtype as dtype.
        """
        self.func = func
        self.dtype = dtype

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Apply `func` on all elements of X.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :returns Xt: array-like, shape (n_samples, n_features)
            The transformed data.
        """
        dtype = self.dtype
        if dtype is None:
            dtype = X.dtype

        vfunc = np.vectorize(self.func, otypes=[dtype])
        return vfunc(X)

class Shaper(BaseEstimator, TransformerMixin):
    """Reshape arrays."""

    def __init__(self, newshape, order="C"):
        """Initialize.

        Parameters
        ----------
        :param newshape: int or tuple
            The new shape of the array.
            See numpy.reshape for further details.

        :param order: {'C', 'F', 'A'}
            The index order.
            See numpy.reshape for further details.
        """
        self.newshape = newshape
        self.order = order

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Reshape X.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :returns Xt: array-like, shape (self.newshape)
            The transformed data.
        """
        return X.reshape(self.newshape, order=self.order)

class PairTransformer(BaseEstimator, TransformerMixin):
    """Apply a transformer on all elements in paired data."""

    def __init__(self, element_transformer, groupby=None):
        """Initialize.

        Parameters
        ----------
        :param element_transformer: transformer
            The transformer to apply on each element.

        :param groupby: callable
            If not None, use ``groupby`` as a hash to apply
            ``element_transformer`` on unique elements only.
        """
        self.element_transformer = element_transformer
        self.groupby = groupby

    def _flatten(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1] // 2
        Xt = X

        # Shortcut, when all elements are distinct
        if self.groupby is None:
            if sp.issparse(Xt):
                Xt = sp.vstack((Xt[:, :n_features],
                                Xt[:, n_features:]))
            else:
                Xt = np.vstack((Xt[:, :n_features],
                                Xt[:, n_features:]))

            return Xt, np.arange(n_samples * 2, dtype=np.int)

        # Group by keys
        groupby = self.groupby
        indices = []        # element index -> first position in X
        key_indices = {}    # key -> first position in X

        for i, element in enumerate(Xt[:, :n_features]):
            key = groupby(element)
            if key not in key_indices:
                key_indices[key] = (i, 0)
            indices.append(key_indices[key])

        for i, element in enumerate(Xt[:, n_features:]):
            key = groupby(element)
            if key not in key_indices:
                key_indices[key] = (i, n_features)
            indices.append(key_indices[key])

        # Select unique elements, from left and right
        left_indices = {}
        right_indices = {}
        key_indices = sorted(key_indices.values())
        j = 0

        for i, start in key_indices:
            if start == 0:
                left_indices[i] = j
                j += 1
        for i, start in key_indices:
            if start == n_features:
                right_indices[i] = j
                j += 1

        if sp.issparse(Xt):
            Xt = sp.vstack((Xt[sorted(left_indices.keys()), :n_features],
                            Xt[sorted(right_indices.keys()), n_features:]))
        else:
            Xt = np.vstack((Xt[sorted(left_indices.keys()), :n_features],
                            Xt[sorted(right_indices.keys()), n_features:]))

        # Map original indices to transformed values
        flat_indices = []

        for i, start in indices:
            if start == 0:
                flat_indices.append(left_indices[i])
            else:
                flat_indices.append(right_indices[i])

        return Xt, flat_indices

    def _repack(self, Xt, indices):
        n_samples = len(indices) // 2

        if sp.issparse(Xt):
            Xt = sp.hstack((Xt[indices[:n_samples]],
                            Xt[indices[n_samples:]]))
        else:
            Xt = np.hstack((Xt[indices[:n_samples]],
                            Xt[indices[n_samples:]]))

        return Xt

    def fit(self, X, y=None):
        """Fit the given transformer on all individual elements in ``X``.

        Rows i in ``X`` are assumed to represent pairs, where
        ``X[i, :n_features]`` and ``X[i, n_features:]`` correspond to their two
        individual elements. Calling ``fit`` trains the given transformer on
        the dataset formed by all these individual elements.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, 2 * n_features)
            Input paired data.

        Returns
        -------
        :returns: self
        """
        Xt, _ = self._flatten(X)
        self.element_transformer.fit(Xt)
        return self

    def transform(self, X):
        """Transform all individual elements in ``X``.

        Rows i in the returned array ``Xt`` represent transformed pairs, where
        ``Xt[i, :n_features_t]`` and ``Xt[i, n_features_t:]`` correspond
        to their two individual transformed elements.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, 2 * n_features)
            Input paired data.

        Returns
        -------
        :returns Xt: array-like, shape (n_samples, 2 * n_features_t
            The transformed data.
        """
        Xt, indices = self._flatten(X)
        Xt = self.element_transformer.transform(Xt)
        Xt = self._repack(Xt, indices)
        return Xt

class CosineSimilarity(BaseEstimator, TransformerMixin):
    """Cosine similarity on paired data."""

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Compute the cosine similarity for all pairs of elements in ``X``.

        Rows i in ``X`` are assumed to represent pairs, where
        ``X[i, :n_features]`` and ``X[i, n_features:]`` correspond to their two
        individual elements. Calling ``transform`` computes the cosine
        similarity between these elements, i.e. that ``Xt[i]`` is the cosine of
        ``X[i, :n_features]`` and ``X[i, n_features:]``.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, 2 * n_features)
            Input paired data.

        Returns
        -------
        :returns Xt: array-like, shape (n_samples, 2 * n_features_prime)
            The transformed data.
        """
        n_samples, n_features_all = X.shape
        n_features = n_features_all // 2
        sparse = sp.issparse(X)

        if sparse and not sp.isspmatrix_csr(X):
            X = X.tocsr()

        X1 = X[:, :n_features]
        X2 = X[:, n_features:]

        if sparse:
            numerator = np.asarray(X1.multiply(X2).sum(axis=1)).ravel()
            norm1 = np.asarray(X1.multiply(X1).sum(axis=1)).ravel()
            norm2 = np.asarray(X2.multiply(X2).sum(axis=1)).ravel()

        else:
            numerator = (X1 * X2).sum(axis=1)
            norm1 = (X1 * X1).sum(axis=1)
            norm2 = (X2 * X2).sum(axis=1)

        denominator = (norm1 ** 0.5) * (norm2 ** 0.5)

        with np.errstate(divide="ignore", invalid="ignore"):
            Xt = numerator / denominator
            Xt[denominator == 0.0] = 0.0

        return Xt.reshape((n_samples, 1))

class JaccardSimilarity(BaseEstimator, TransformerMixin):
    """Jaccard similarity on paired data.

    The Jaccard similarity of two elements in a pair is defined as the
    ratio between the size of their intersection and the size of their
    union.
    """

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Compute the Jaccard similarity for all pairs of elements in ``X``.

        Rows i in ``X`` are assumed to represent pairs, where
        ``X[i, :n_features]`` and ``X[i, n_features:]`` correspond to their two
        individual elements, each representing a set. Calling ``transform``
        computes the Jaccard similarity between these sets, i.e. such that
        ``Xt[i]`` is the Jaccard similarity of ``X[i, :n_features]`` and
        ``X[i, n_features:]``.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :returns: Xt array-like, shape (n_samples, 1)
            The transformed data.
        """
        n_samples, n_features_all = X.shape
        n_features = n_features_all // 2

        X = binarize(X)
        X1 = X[:, :n_features]
        X2 = X[:, n_features:]

        sparse = sp.issparse(X)

        if sparse and not sp.isspmatrix_csr(X):
            X = X.tocsr()

        if sparse:
            if X.data.sum() == 0:
                return np.zeros((n_samples, 1))

            numerator = np.asarray(X1.multiply(X2).sum(axis=1)).ravel()

            X_sum = X1 + X2
            X_sum.data[X_sum.data != 0.] = 1
            M = X_sum.sum(axis=1)
            A = M.getA()
            denominator = A.reshape(-1,)

        else:
            if len(X[X.nonzero()]) == 0.:
                return np.zeros((n_samples, 1))

            numerator = (X1 * X2).sum(axis=1)

            X_sum = X1 + X2
            X_sum[X_sum.nonzero()] = 1
            denominator = X_sum.sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            Xt = numerator / denominator
            Xt[np.where(denominator == 0)[0]] = 0.

        return np.array(Xt).reshape(-1, 1)

def _jaro_winkler_similarity(x, y):
    if len(x) <= 1 or len(y) <= 1:
        return -1.

    return jellyfish.jaro_winkler(x, y)

def _character_equality(x, y):
    if x != y:
        return 0.
    elif x == "":
        return 0.5
    else:
        return 1.

class StringDistance(BaseEstimator, TransformerMixin):
    """Distance between strings on paired data.

    It can be fed with a custom similarity function. By default jaro winkler is
    used.
    """

    def __init__(self, similarity_function="jaroWin"):
        """Initialize the transformer.

        Parameters
        ----------
        :param similarity_function: function (string, string) -> float
            Function that will evaluate similarity of the paired data.
        """
        if similarity_function == "jaroWin":
            self.similarity_function = _jaro_winkler_similarity
        elif similarity_function == "character_equality":
            self.similarity_function = _character_equality

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Compute string similarity.

        Rows i in ``X`` are assumed to represent pairs, where
        ``X[i, :n_features]`` and ``X[i, n_features:]`` correspond to their
        individual elements, each representing a string.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :returns: Xt array-like, shape (n_samples, 1)
            The transformed data.
        """
        X1, X2 = np.split(X, 2, axis=1)

        vectorized = np.vectorize(self.similarity_function)
        n_samples = X1.shape[0]

        val = vectorized(X1, X2)
        return val.reshape((n_samples, 1))


""" Clustering Method """

class ClusteringMethods(BaseEstimator, ClusterMixin):
    
    """
    Wrapper for various clustering implementations.

    Attributes
    ----------
    labels_ : ndarray, shape (n_samples,)
        Array of labels assigned to the input data.

    linkage_ : ndarray
        The linkage matrix.
        
    """

    def __init__(
                 self, method              = "single", 
                 affinity                  = "euclidean",
                 n_clusters                = None, 
                 clustering_algorithm      = None,
                 threshold                 = None, 
                 criterion                 = "distance",
                 depth                     = 2, 
                 R                         = None, 
                 monocrit                  = None, 
                 unsupervised_scoring      = None,
                 supervised_scoring        = None, 
                 scoring_data              = None,
                 best_threshold_precedence = True
                 ):
              
         
        self.method                    = method
        self.affinity                  = affinity
        self.n_clusters                = n_clusters
        self.clustering_algorithm      = clustering_algorithm
        self.threshold                 = threshold
        self.criterion                 = criterion
        self.depth                     = depth
        self.R                         = R
        self.monocrit                  = monocrit
        self.unsupervised_scoring      = unsupervised_scoring
        self.supervised_scoring        = supervised_scoring
        self.scoring_data              = scoring_data
        self.best_threshold_precedence = best_threshold_precedence

    def fit(self, X, y=None):
        """
        Perform hierarchical clustering on input data.

        :param X: array-like, shape (n_samples, n_features) or
                  (n_samples, n_samples)
            Input data, as an array of samples or as a distance matrix if
            affinity == 'precomputed'.

        :param y: array-like, shape (n_samples, )
            Input labels, in case of (semi-)supervised clustering.
            Labels equal to -1 stand for unknown labels.

        :returns: self
        """
        X         = np.array(X)
        X_raw     = X
        n_samples = X.shape[0]

        clusterer_names = ["spectral", "db", "kmeans", "agg"]
        
        # Build linkage matrix
        if self.affinity == "precomputed" or callable(self.affinity):
            
            if callable(self.affinity):
                X = self.affinity(X, clustering_algorithm=self.clustering_algorithm)

            X_affinity = X

            if self.clustering_algorithm == "hier":
                self.linkage_ = hac.linkage(X, method=self.method)
                
                # Estimate threshold in case of semi-supervised or unsupervised
                # As default value we use the highest so we obtain only 1 cluster.
                # best_threshold = (self.linkage_[-1, 2] if self.threshold is None else self.threshold)
              
            elif self.clustering_algorithm == "spectral":
                self.spectral= SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed', random_state=0).fit(X)
            
            elif self.clustering_algorithm == "db":
                self.db = DBSCAN().fit(X)

            elif self.clustering_algorithm == "kmeans":
                self.kmeans= KMeans(n_clusters=self.n_clusters, n_init=10, max_iter=300, random_state=0).fit(X)
            
            elif self.clustering_algorithm == "agg":
                self.agg = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='average').fit(X)

            else:
                raise ValueError('Invalid name for clustering method!') 

        else:
            X_affinity = None
            self.linkage_ = hac.linkage(X,
                                        method=self.method,
                                        metric=self.affinity)

            # Estimate threshold in case of semi-supervised or unsupervised
            # As default value we use the highest so we obtain only 1 cluster.
            best_threshold = (self.linkage_[-1, 2] if self.threshold is None else self.threshold)

        n_clusters           = self.n_clusters
        supervised_scoring   = self.supervised_scoring
        unsupervised_scoring = self.unsupervised_scoring
        ground_truth = (y is not None) and np.any(np.array(y) != -1)
        scoring = supervised_scoring is not None or             unsupervised_scoring is not None

        if self.clustering_algorithm == "hier":
            
            if n_clusters is None and scoring:
                                
                if self.threshold:
                    
                    threshold = self.threshold
                    
                    labels    = hac.fcluster(
                                             self.linkage_,
                                             threshold,
                                             criterion = self.criterion,
                                             depth     = self.depth,
                                             R         = self.R,
                                             monocrit  = self.monocrit
                                             )
                
                    precision = ''
                    recall    = ''
                    fscore    = ''
                    
                    if ground_truth and supervised_scoring is not None:
                        train = (y != -1)

                        if self.scoring_data == "raw":
                            precision, recall, fscore = supervised_scoring(X_raw, y[train], labels[train])
                        elif self.scoring_data == "affinity":
                            precision, recall, fscore = supervised_scoring(X_affinity, y[train], labels[train])
                        else:
                            precision, recall, fscore = supervised_scoring(y[train], labels[train])                        

                    elif unsupervised_scoring is not None:
                        
                        if self.scoring_data == "raw":
                            precision, recall, fscore = unsupervised_scoring(X_raw, labels)
                        elif self.scoring_data == "affinity":
                            precision, recall, fscore = unsupervised_scoring(X_affinity, labels)
                        else:
                            precision, recall, fscore = unsupervised_scoring(labels)
                                
                    
                    best_threshold = threshold
                    best_precision = precision
                    best_recall    = recall
                    best_score     = fscore
                    
                else:
                    
                    best_score = -np.inf
                    thresholds = np.concatenate(([0], self.linkage_[:, 2], [self.linkage_[-1, 2]]))
                
                    for i in range(len(thresholds) - 1):
                        t1, t2 = thresholds[i:i + 2]
                        threshold = (t1 + t2) / 2.0

                        labels    = hac.fcluster(
                                                 self.linkage_,
                                                 threshold,
                                                 criterion = self.criterion,
                                                 depth     = self.depth,
                                                 R         = self.R,
                                                 monocrit  = self.monocrit
                                                 )
                
                        if ground_truth and supervised_scoring is not None:
                            train = (y != -1)

                            if self.scoring_data == "raw":
                                precision, recall, fscore = supervised_scoring(X_raw, y[train], labels[train])
                            elif self.scoring_data == "affinity":
                                precision, recall, fscore = supervised_scoring(X_affinity, y[train], labels[train])
                            else:
                                precision, recall, fscore = supervised_scoring(y[train], labels[train])                        

                        elif unsupervised_scoring is not None:
                        
                            if self.scoring_data == "raw":
                                precision, recall, fscore = unsupervised_scoring(X_raw, labels)
                            elif self.scoring_data == "affinity":
                                precision, recall, fscore = unsupervised_scoring(X_affinity, labels)
                            else:
                                precision, recall, fscore = unsupervised_scoring(labels)
                        else:
                            break

                        if fscore >= best_score:
                            best_score     = fscore
                            best_threshold = threshold
                            best_precision = precision
                            best_recall    = recall
            
            else:
                raise ValueError("n_clusters must be None for hierarchical clustering")
                
            # output: samples, threshold, precision, recall, f1
            self.n_samples_ = n_samples
            self.best_threshold_ = best_threshold

            if ground_truth:       
                self.best_precision_ = best_precision
                self.best_recall_    = best_recall
                self.best_fscore_    = best_score
    
                self.best_scores_ = (self.best_threshold_, self.best_precision_, self.best_recall_, self.best_fscore_)
            else:
                self.best_scores_ = None
          
        elif self.clustering_algorithm in clusterer_names:
            
            if self.clustering_algorithm == "db":
                db     = self.db
                labels = db.labels_

            elif self.clustering_algorithm == "spectral":
                spectral = self.spectral
                labels   = spectral.labels_

            elif self.clustering_algorithm == "kmeans":
                kmeans = self.kmeans
                labels = kmeans.labels_
            elif self.clustering_algorithm == "agg":
                agg    = self.agg
                labels = agg.labels_

            if scoring:
                                
                if ground_truth and supervised_scoring is not None:
                    
                    train = (y != -1)

                    if self.scoring_data == "raw":
                        precision, recall, fscore = supervised_scoring(X_raw, y[train], labels[train])
                    elif self.scoring_data == "affinity":
                        precision, recall, fscore = supervised_scoring(X_affinity, y[train], labels[train])
                    else:
                        precision, recall, fscore = supervised_scoring(y[train], labels[train])                        
            
                elif unsupervised_scoring is not None:
                    
                    if self.scoring_data == "raw":
                        precision, recall, fscore = unsupervised_scoring(X_raw, labels)
                    elif self.scoring_data == "affinity":
                        precision, recall, fscore = unsupervised_scoring(X_affinity, labels)
                    else:
                        precision, recall, fscore = unsupervised_scoring(labels)
            
            self.n_samples_ = n_samples

            if ground_truth:
                self.best_precision_ = precision
                self.best_recall_    = recall
                self.best_fscore_    = fscore 
            
                self.best_scores_ = (self.best_precision_, self.best_recall_, self.best_fscore_)
            else:
                self.best_scores_ = None

        return self

    @property
    def labels_(self):
        """
           Compute the labels assigned to the input data.

        """
        n_clusters = self.n_clusters

        if n_clusters is not None:
            if n_clusters < 1 or n_clusters > self.n_samples_:
                raise ValueError("n_clusters must be within [1; n_samples].")

            else:
                if   self.clustering_algorithm == "hier":
                    thresholds = np.concatenate(([0], self.linkage_[:, 2], [self.linkage_[-1, 2]]))

                    for i in range(len(thresholds) - 1):
                        t1, t2 = thresholds[i:i + 2]
                        threshold = (t1 + t2) / 2.0
                        labels = hac.fcluster(
                                              self.linkage_,
                                              threshold,
                                              criterion = self.criterion,
                                              depth     = self.depth,
                                              R         = self.R,
                                              monocrit  = self.monocrit
                                              )
                        
                        if len(np.unique(labels)) <= n_clusters:
                            _, labels = np.unique(labels, return_inverse=True)
                            return labels
                    
                elif self.clustering_algorithm == "db":
                    db = self.db
                    labels = db.labels_
                    if len(np.unique(labels)) <= n_clusters:
                        _, labels = np.unique(labels, return_inverse=True)
                        return labels

                elif self.clustering_algorithm == "spectral":
                    spectral = self.spectral
                    labels = spectral.labels_
                    if len(np.unique(labels)) <= n_clusters:
                        _, labels = np.unique(labels, return_inverse=True)
                        return labels

                elif self.clustering_algorithm == "kmeans":
                    kmeans = self.kmeans
                    labels = kmeans.labels_
                    if len(np.unique(labels)) <= n_clusters:
                        _, labels = np.unique(labels, return_inverse=True)
                        return labels               

        else:
            if   self.clustering_algorithm == "hier":
                threshold = self.threshold

                if self.best_threshold_precedence:
                    threshold = self.best_threshold_

                labels = hac.fcluster(
                                      self.linkage_,
                                      threshold,
                                      criterion = self.criterion,
                                      depth     = self.depth,
                                      R         = self.R,
                                      monocrit  = self.monocrit
                                      )
      
            elif self.clustering_algorithm == "db":
                db    = self.db
                abels = db.labels_   
            
            elif self.clustering_algorithm == "spectral":
                spectral = self.spectral
                labels   = spectral.labels_ 
                
            elif self.clustering_algorithm == "kmeans":
                kmeans = self.kmeans
                labels = kmeans.labels_                     
                              
            _, labels = np.unique(labels, return_inverse=True)
            
            return labels
    
    def scores_(self):     
        
        if self.best_scores_:
            self.scores = self.best_scores_
        else:
            self.scores = None
        
        return self.scores

class ClusteringWithoutBlocking(BaseEstimator, ClusterMixin):
    
    
    """Wrapper for clustering implementation with no blocking applied.

    Attributes
    ----------
    labels_ : ndarray, shape (n_samples,)
        Array of labels assigned to the input data.

    linkage_ : ndarray
        The linkage matrix.
    """

    def __init__(
                 self, method           = "single",
                 affinity               = "euclidean",
                 n_clusters             = None,
                 clustering_algorithm   = None,
                 result_per_block = None,
                 unsupervised_scoring   = None,
                 supervised_scoring     = None,
                 scoring_data           = None
                 ):
              
         
        self.method                 = method
        self.affinity               = affinity
        self.n_clusters             = n_clusters
        self.clustering_algorithm   = clustering_algorithm
        self.result_per_block = result_per_block 
        self.unsupervised_scoring   = unsupervised_scoring
        self.supervised_scoring     = supervised_scoring
        self.scoring_data           = scoring_data

    def fit(self, X, y=None):
        """Perform clustering on input data.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features) or
                  (n_samples, n_samples)
            Input data, as an array of samples or as a distance matrix if
            affinity == 'precomputed'.

        :param y: array-like, shape (n_samples, )
            Input labels, in case of (semi-)supervised clustering.
            Labels equal to -1 stand for unknown labels.

        Returns
        -------
        :returns: self
        """
        X     = np.array(X)
        X_raw = X

        # print("No of Xt: ", len(X_raw))
        
        n_samples = X.shape[0]
        
        # Build linkage matrix
        if self.affinity == "precomputed" or callable(self.affinity):
            
            if callable(self.affinity):
                X = self.affinity(X, clustering_algorithm=self.clustering_algorithm)

            X_affinity = X
               
            if   self.clustering_algorithm == "db":
                self.db = DBSCAN().fit(X)

            elif self.clustering_algorithm == "spectral":
                self.spectral= SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed', random_state=0).fit(X)

            elif self.clustering_algorithm == "kmeans":
                self.kmeans= KMeans(n_clusters=self.n_clusters, n_init=10, max_iter=300, random_state=0).fit(X)

            elif self.clustering_algorithm == "agg":
                self.agg = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.method).fit(X)

        else:
            X_affinity = None

        n_clusters           = self.n_clusters
        supervised_scoring   = self.supervised_scoring
        unsupervised_scoring = self.unsupervised_scoring
        ground_truth = (y is not None) and np.any(np.array(y) != -1)
        scoring      = supervised_scoring is not None or             unsupervised_scoring is not None
            
        if   self.clustering_algorithm == "db":
            db     = self.db
            labels = db.labels_
        elif self.clustering_algorithm == "spectral":
            spectral = self.spectral
            labels   = spectral.labels_
        elif self.clustering_algorithm == "kmeans":
            kmeans = self.kmeans
            labels = kmeans.labels_
        elif self.clustering_algorithm == "agg":
            agg    = self.agg
            labels = agg.labels_
        
        if scoring:
            if ground_truth and supervised_scoring is not None:
                
                train = (y != -1)
                precision, recall, fscore = supervised_scoring(y[train], labels[train])                        
            
            elif unsupervised_scoring is not None:

                precision, recall, fscore = unsupervised_scoring(labels)
            
        self.n_samples_ = n_samples
        self.precision_ = precision
        self.recall_    = recall
        self.f_score_   = fscore

        metric_results = "{}|{:.4f}|{:.4f}|{:.4f}".format(
                                                          self.n_samples_,
                                                          self.precision_,
                                                          self.recall_,
                                                          self.f_score_
                                                          )

        print("\nn_samples|precision|recall|F1")
        print(metric_results)
        print("\n")


        if self.result_per_block is not None:
            with open(self.result_per_block, 'a') as f:
                f.write("Samples|Precision|Recall|F1\n")
                f.write(metric_results + '\n')


        return self

    @property
    def labels_(self):
        """
           Compute the labels assigned to the input data.

        """
        n_clusters = self.n_clusters

        if n_clusters is not None:
            if n_clusters < 1 or n_clusters > self.n_samples_:
                raise ValueError("n_clusters must be within [1; n_samples].")

            else: 
                if   self.clustering_algorithm == "db":
                    db     = self.db
                    labels = db.labels_ 
                elif self.clustering_algorithm == "spectral":
                    spectral = self.spectral
                    labels   = spectral.labels_ 
                elif self.clustering_algorithm == "kmeans":
                    kmeans = self.kmeans
                    labels = kmeans.labels_ 
                elif self.clustering_algorithm == "agg":
                    agg    = self.agg
                    labels = agg.labels_

                if len(np.unique(labels)) <= n_clusters:
                    _, labels = np.unique(labels, return_inverse=True)
                    return labels                

        else:                     
            if   self.clustering_algorithm == "db":
                db    = self.db
                abels = db.labels_ 
            elif self.clustering_algorithm == "spectral":
                spectral = self.spectral
                labels  = spectral.labels_
            elif self.clustering_algorithm == "kmeans":
                kmeans = self.kmeans
                labels = kmeans.labels_                     
            elif self.clustering_algorithm == "agg":
                agg    = self.agg
                labels = agg.labels_
                              
            _, labels = np.unique(labels, return_inverse=True)
            
            return labels

class _SingleClustering(BaseEstimator, ClusterMixin):
    def fit(self, X, y=None):
        self.labels_ = block_single(X)
        return self

    def partial_fit(self, X, y=None):
        self.labels_ = block_single(X)
        return self

    def predict(self, X):
        return block_single(X)

def _parallel_fit(fit_, partial_fit_, estimator, verbose, data_queue, result_queue):

    """Run clusterer's fit function."""
    # Status can be one of: 'middle', 'end'
    # 'middle' means that there is a block to compute and the process should
    # continue
    # 'end' means that the process should finish as all the data was sent
    # by the main process
    status, block, existing_clusterer = data_queue.get()

    while status != 'end':

        b, X, y = block

        if len(X) == 1:
            clusterer = _SingleClustering()
        elif existing_clusterer and partial_fit_ and not fit_:
            clusterer = existing_clusterer
        else:
            clusterer = clone(estimator)

        if verbose > 1:
            # print("Clustering %d samples on block '%s'..." % (len(X), b))
            print("Block '%s': %d samples" % (b, len(X)))

        if fit_ or not hasattr(clusterer, "partial_fit"):
            try:
                clusterer.fit(X, y=y)
            except TypeError:
                clusterer.fit(X)
        elif partial_fit_:
            try:
                clusterer.partial_fit(X, y=y)
            except TypeError:
                clusterer.partial_fit(X)

        result_queue.put((b, clusterer))
        status, block, existing_clusterer = data_queue.get()

    data_queue.put(('end', None, None))
    return

def _single_fit(fit_, partial_fit_, estimator, verbose, data):
    """Run clusterer's fit function."""
                
    block, existing_clusterer = data
    b, X, y = block  
    
    if len(X) == 1:
        clusterer = _SingleClustering()
    elif existing_clusterer and partial_fit_ and not fit_:
        clusterer = existing_clusterer
    else:
        clusterer = clone(estimator)

    # if verbose > 1:
        # print("Block '%s': %d samples" % (b, len(X)))

    if fit_ or not hasattr(clusterer, "partial_fit"):
        try:
            clusterer.fit(X, y=y)
        except TypeError:
            clusterer.fit(X)
    elif partial_fit_:
        try:
            clusterer.partial_fit(X, y=y)
        except TypeError:
            clusterer.partial_fit(X)

    return (b, len(X), clusterer)

def block_single(X):
    """Block the signatures into only one block.

    Parameters
    ----------
    :param X: numpy array
        Array of singletons of dictionaries.

    Returns
    -------
    :returns: numpy array
        Array with ids of the blocks. As there is only one block, every element
        equals zero.
    """
    return np.zeros(len(X), dtype=np.int)

class BlockClustering(BaseEstimator, ClusterMixin):
    """Implements blocking for clustering estimators.

    Meta-estimator for grouping samples into blocks, within each of which
    a clustering base estimator is fit. This allows to reduce the cost of
    pairwise distance computation from O(N^2) to O(sum_b N_b^2), where
    N_b <= N is the number of samples in block b.

    Attributes
    ----------
    labels_ : ndarray, shape (n_samples,)
        Array of labels assigned to the input data.
        if partial_fit is used instead of fit, they are assigned to the
        last batch of data.

    blocks_ : ndarray, shape (n_samples,)
        Array of keys mapping input data to blocks.
    """

    def __init__(
                 self,
                 affinity           = None,
                 blocking           = "single",
                 base_estimator     = None,
                 verbose            = 1,
                 result_per_block = None,
                 n_jobs=1
                 ):

        """Initialize.

        Parameters
        ----------
        :param affinity: string or None
            If affinity == 'precomputed', then assume that X is a distance
            matrix.

        :param blocking: string or callable, default "single"
            The blocking strategy, for mapping samples X to blocks.
            - "single": group all samples X[i] into the same block;
            - "precomputed": use `blocks[i]` argument (in `fit`, `partial_fit`
              or `predict`) as a key for mapping sample X[i] to a block;
            - callable: use blocking(X)[i] as a key for mapping sample X[i] to
              a block.

        :param base_estimator: estimator
            Clustering estimator to fit within each block.

        :param verbose: int, default=0
            Verbosity of the fitting procedure.

        :param n_jobs: int
            Number of processes to use.
        """
        self.affinity           = affinity
        self.blocking           = blocking
        self.base_estimator     = base_estimator
        self.verbose            = verbose
        self.result_per_block = result_per_block
        self.n_jobs             = n_jobs
        

    def _validate(self, X, blocks):
        """Validate hyper-parameters and input data."""
        if self.blocking == "single":
            blocks = block_single(X)
        elif self.blocking == "precomputed":
            if blocks is not None and len(blocks) == len(X):
                blocks = column_or_1d(blocks).ravel()
            else:
                raise ValueError("Invalid value for blocks. When "
                                 "blocking='precomputed', blocks needs to be "
                                 "an array of size len(X).")
        elif callable(self.blocking):
            blocks = self.blocking(X)
        else:
            raise ValueError("Invalid value for blocking. Allowed values are "
                             "'single', 'precomputed' or callable.")

        return X, blocks

    def _blocks(self, X, y, blocks):
        """Chop the training data into smaller chunks.

        A chunk is demarcated by the corresponding block. Each chunk contains
        only the training examples relevant to given block and a clusterer
        which will be used to fit the data.

        Returns
        -------
        :returns: generator
            Quadruples in the form of ``(block, X, y, clusterer)`` where
            X and y are the training examples for given block and clusterer is
            an object with a ``fit`` method.
        """
        unique_blocks = np.unique(blocks)

        for b in unique_blocks:
            mask = (blocks == b)
            X_mask = X[mask, :]
            if y is not None:
                y_mask = y[mask]
            else:
                y_mask = None
            if self.affinity == "precomputed":
                X_mask = X_mask[:, mask]

            yield (b, X_mask, y_mask)

    def _fit(self, X, y, blocks):
        """Fit base clustering estimators on X."""
        self.blocks_ = blocks

        if self.n_jobs == 1:
            blocks_computed = 0
            blocks_all      = len(np.unique(blocks))
            
            self.blocks_scores  = {}
            self.blocks_samples = {}

            for block in self._blocks(X, y, blocks):
                if self.partial_fit_ and block[0] in self.clusterers_:
                    data = (block, self.clusterers_[block[0]])
                else:
                    data = (block, None)

                b, sampleCount, clusterer = _single_fit(
                                                        self.fit_,
                                                        self.partial_fit_,
                                                        self.base_estimator,
                                                        self.verbose,
                                                        data
                                                        )
                    
                if clusterer:                  
                    self.clusterers_[b] = clusterer
                    
                    ## Ignore a block with only 1 instance included
                    if sampleCount > 1 and self.clusterers_[b].scores_() is not None:
                        ## extract scores for each block: threshold, precision, recall, f1-score
                        self.blocks_scores[b] = self.clusterers_[b].scores_()         
                    
                    self.blocks_samples[b] = sampleCount
                blocks_computed += 1
            
            if self.blocks_scores:
                
                ## Print out scores for each block 
                ## compute the mean and standard deviation scores for all blocks 
                thr_list = [] # <- threshold
                pre_list = [] # <- precision
                rec_list = [] # <- recall
                fsc_list = [] # <- f1-score
                scr_list = [] # <- score
            
                for block, values in self.blocks_scores.items():
                
                    threshold = values[0]
                    precision = values[1]
                    recall    = values[2]
                    f1_score  = values[3]
                    samples_per_block = self.blocks_samples[block]
                    format_style = "{}|{}|{:.4f}|{:.4f}|{:.4f}|{:.4f}"
                    metric_results = format_style.format(
                                                         block,
                                                         samples_per_block,
                                                         threshold,
                                                         precision,
                                                         recall,
                                                         f1_score
                                                        )
                
                    thr_list.append(threshold)
                    pre_list.append(precision)
                    rec_list.append(recall)
                    fsc_list.append(f1_score)
                    scr_list.append(metric_results)
        
                thr_array = np.array(thr_list)
                pre_array = np.array(pre_list)
                rec_array = np.array(rec_list)
                fsc_array = np.array(fsc_list)
            
                ## metrics scores for each block
                metrics_out = "\n".join(scr_list)
                if self.verbose == 1:
                    print("\nblock name|instances|threshold|precision|recall|f1")
                    print(metrics_out)

                if self.result_per_block:
                    with open(self.result_per_block, 'w') as f:
                        f.write("block name|instances|threshold|precision|recall|f1\n")
                        f.write(metrics_out)
            
                ## Mean and standard deviation for all blocks
                #print("    Mean(SD) evaluation scores for all blocks: ")
                #print("    threshold|precision|recall|f-score")
                #block_score_style = "    {:.4f}({:.4f})|{:.4f}({:.4f})|{:.4f}({:.4f})|{:.4f}({:.4f})"
                #print(
                #      block_score_style.format(
                #                               thr_array.mean(), thr_array.std(), 
                #                               pre_array.mean(), pre_array.std(), 
                #                               rec_array.mean(), rec_array.std(), 
                #                               fsc_array.mean(), fsc_array.std()
                #                              ), "\n"
                #      )
                
                ## avreage threshold score
                self.avg_thres_ = round(thr_array.mean(), 4)
                
    
        else:
            try:
                from multiprocessing import SimpleQueue
            except ImportError:
                from multiprocessing.queues import SimpleQueue

            # Here the blocks will be passed to subprocesses
            data_queue = SimpleQueue()
            # Here the results will be passed back
            result_queue = SimpleQueue()

            for x in range(self.n_jobs):
                import multiprocessing as mp
                processes = []

                processes.append(
                                 mp.Process(
                                            target=_parallel_fit,
                                            args=(
                                                  self.fit_,
                                                  self.partial_fit_,
                                                  self.base_estimator,
                                                  self.verbose,
                                                  data_queue,
                                                  result_queue
                                                  )
                                            )
                                 )
                processes[-1].start()

            # First n_jobs blocks are sent into the queue without waiting
            # for the results. This variable is a counter that takes care of
            # this.
            presend = 0
            blocks_computed = 0
            blocks_all = len(np.unique(blocks))

            for block in self._blocks(X, y, blocks):
                if presend >= self.n_jobs:
                    b, clusterer = result_queue.get()
                    blocks_computed += 1
                    if clusterer:
                        self.clusterers_[b] = clusterer
                else:
                    presend += 1
                if self.partial_fit_:
                    if block[0] in self.clusterers_:
                        data_queue.put(('middle', block, self.clusterers_[b]))
                        continue

                data_queue.put(('middle', block, None))

            # Get the last results and tell the subprocesses to finish
            for x in range(self.n_jobs):
                if blocks_computed < blocks_all:
#                     print("%s blocks computed out of %s" % (blocks_computed,
#                                                             blocks_all))
                    b, clusterer = result_queue.get()
                    blocks_computed += 1
                    if clusterer:
                        self.clusterers_[b] = clusterer

            data_queue.put(('end', None, None))

            time.sleep(1)

        return self

    def fit(self, X, y=None, blocks=None):
        """Fit individual base clustering estimators for each block.

        Parameters
        ----------
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
                  or (n_samples, n_samples)
            Input data, as an array of samples or as a distance matrix if
            affinity == 'precomputed'.

        :param y: array-like, shape (n_samples, )
            Input labels, in case of (semi-)supervised clustering.
            Labels equal to -1 stand for unknown labels.

        :param blocks: array-like, shape (n_samples, )
            Block labels, if `blocking == 'precomputed'`.

        Returns
        -------
        :returns: self
        """
        # Validate parameters
        X, blocks = self._validate(X, blocks)

        # Reset attributes
        self.clusterers_ = {}
        self.fit_, self.partial_fit_ = True, False

        return self._fit(X, y, blocks)

    def partial_fit(self, X, y=None, blocks=None):
        """Resume fitting of base clustering estimators, for each block.

        This calls `partial_fit` whenever supported by the base estimator.
        Otherwise, this calls `fit`, on given blocks only.

        Parameters
        ----------
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
                  or (n_samples, n_samples)
            Input data, as an array of samples or as a distance matrix if
            affinity == 'precomputed'.

        :param y: array-like, shape (n_samples, )
            Input labels, in case of (semi-)supervised clustering.
            Labels equal to -1 stand for unknown labels.

        :param blocks: array-like, shape (n_samples, )
            Block labels, if `blocking == 'precomputed'`.

        Returns
        -------
        :returns: self
        """
        # Validate parameters
        X, blocks = self._validate(X, blocks)

        # Set attributes if first call
        if not hasattr(self, "clusterers_"):
            self.clusterers_ = {}

        self.fit_, self.partial_fit_ = False, True

        return self._fit(X, y, blocks)

    def predict(self, X, blocks=None):
        """Predict data.

        Parameters
        ----------
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.

        :param blocks: array-like, shape (n_samples, )
            Block labels, if `blocking == 'precomputed'`.

        Returns
        -------
        :returns: array-like, shape (n_samples)
            The labels.
        """
        # Validate parameters
        X, blocks = self._validate(X, blocks)

        # Predict
        labels = -np.ones(len(X), dtype=np.int)
        offset = 0

        for b in np.unique(blocks):
            # Predict on the block, if known
            if b in self.clusterers_:
                mask                = (blocks == b)
                clusterer           = self.clusterers_[b]
                pred                = np.array(clusterer.predict(X[mask]))
                pred[(pred != -1)] += offset
                labels[mask]        = pred
                offset             += np.max(clusterer.labels_) + 1

        return labels

    @property
    def labels_(self):
        """Compute the labels assigned to the input data.

        Note that labels are computed on-the-fly.
        """
        labels = -np.ones(len(self.blocks_), dtype=np.int)
        offset = 0

        for b in self.clusterers_:
            mask = (self.blocks_ == b)
            clusterer    = self.clusterers_[b]
            pred         = np.array(clusterer.labels_)
            pred[(pred != -1)] += offset
            labels[mask] = pred
            offset      += np.max(clusterer.labels_) + 1

        return labels

    @property
    def avg_threshold_(self):
        """Assign average threshold score """
        
        if self.avg_thres_:
            avg_score = self.avg_thres_
        else:
            avg_score = None
        
        return avg_score
### The end of line ###