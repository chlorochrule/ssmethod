# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from lpproj import LocalityPreservingProjection


class SubspaceClassifier():
    """A subspace method classifier.
    
    Parameters
    ----------
    n_components : integer
        The dimension of subspace in subspace method classifier.  

    n_estimators : integer, optional (default=1)
        The number of estimators for ensemble subspace method.

    max_bases : integer or None, optional (default=None)
        Max number of bases. Components are retrieved from these bases
        by restoration extraction. If `None`, same as `n_components`.

    basis_type : string, optional (default='pca')
        The method to select bases.

        - If 'pca', then bases are decided by PCA.
        - If 'laplacian', then bases are decided by LLP.
        - If 'custom', then bases are decided by `basis_func`.

    basis_func : function, optional (default=None)
        The customized method to calculate components of each class. This 
        option is valid only `basis_type='custom'`.

    Attributes
    ----------
    accuracy_ : float (or None)
        The accuracy of prediction. If the `predict` method does not get 
        `y` parameter, `accuracy_` is None.
    """ 
    def __init__(self, n_components, n_estimators=1, max_bases=None,
                 basis_type='pca', basis_func=None):
        self.n_components = n_components
        self.n_estimators = n_estimators
        self.max_bases = max_bases
        self.basis_type = basis_type
        self.basis_func = basis_func
        self.bases = None

    def fit(self, X, y):
        """fit
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input training data.

        y : array-like, shape = [n_samples]
            The answer label of training data.
        """
        X = check_array(X)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        self.y_ = self.label_encoder.transform(y)
        X_ = []
        n_classes = self.label_encoder.classes_.shape[0]

        for i in range(n_classes):
            X_.append(X[self.y_==i, :])

        if self.max_bases is not None:
            max_bases = self.max_bases
        else:
            max_bases = self.n_components

        self.bases = np.empty((n_classes, max_bases, X.shape[1]),
                              dtype=np.float)

        for i in range(n_classes):
            if self.basis_type == 'pca':
                pca = PCA(n_components=max_bases)
                pca.fit(X_[i])
                self.bases[i, :, :] = pca.components_
            elif self.basis_type == 'laplacian':
                lpp = LocalityPreservingProjection(n_components=max_bases)
                lpp.fit(X_[i])
                self.bases[i, :, :] = lpp.projection_.T
            elif self.basis_type == 'custom':
                self.bases[i, :, :] = self.basis_func(X_[i])
            else:
                raise ValueError("`basis_type` is not valid")

        return self

    def predict(self, X, y=None):
        """predict
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input predictors. 

        y : array-like, shape = [n_samples]
            The answer labels.

        Returens
        --------
        pred : array-like, shape = [n_samples]
            The predicted labels.
        """
        # TODO: enable ensemble
        X = check_array(X)

        UX = np.dot(self.bases, X.T)
        UX2_sum = np.sum(UX ** 2, axis=1)
        y_ = np.argmax(UX2_sum, axis=0)
        pred = self.label_encoder.inverse_transform(y_)
        if y is not None:
            y = self.label_encoder.transform(y)
            self.accuracy_ = np.sum(y==pred) / pred.shape[0]
        else:
            self.accuracy_ = None
        return pred
        
