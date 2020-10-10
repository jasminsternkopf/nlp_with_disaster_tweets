import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

"""
All neccessary functions for SiPLS and LSIPLS and SIPLS and LSIPLS themselves as a Classifier (SIPLS) or Transformer (LSIPLS) are defined
Because the data should be centered (columns should have mean zero) and the rows should be normalized after the projection into a lowerdimensional space (not for SIPLS), a Centerer and a Row_Normalizer are defined
"""


def LSIPLS_fit(X, y, no_of_latent_vars):
    E = X
    f = y - y.mean()
    n = X.shape[1]
    Xi = np.zeros((n, no_of_latent_vars))
    for k in range(no_of_latent_vars):
        Xi[:, k] = (E.T @ f) / np.linalg.norm(E.T @ f)
        t = E @ Xi[:, k]
        p = (E.T @ t) / np.inner(t, t)
        q = np.inner(f, t) / np.inner(t, t)
        E = E - np.outer(t, p)
        f = f - q * t
    return Xi


class Centerer(TransformerMixin):

    def __init__(self):
        self.mean = None

    def fit(self, X, y=None):
        self.mean = X.mean(axis=0)
        return self

    def transform(self, X, y=None):
        return X - self.mean


class Row_Normalizer(TransformerMixin):

    def __init__(self, norm=None):
        self.norm = None

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        m = X.shape[0]
        for i in range(m):
            X[i, :] = X[i, :] / np.linalg.norm(X[i, :], ord=self.norm)
        return X


class SIPLS(BaseEstimator):
    def __init__(self, no_of_latent_vars=2):
        self.no_of_latent_vars = no_of_latent_vars

    def fit(self, X, y):
        E = X
        y2 = -y.copy()
        y2 = y2 + 1
        F = np.c_[y, y2]
        m, n = X.shape
        x_centroid = 1 / m * X.sum(axis=0)
        y_centroid = 1 / m * F.sum(axis=0)
        Xi = np.zeros((n, self.no_of_latent_vars))
        Omega = np.zeros((2, self.no_of_latent_vars))
        Principale_components = np.zeros((n, self.no_of_latent_vars))
        tol = 1e-10
        for k in range(self.no_of_latent_vars):
            f = y
            while np.abs(np.linalg.norm(Xi[:, k] - (E.T @ f / np.inner(f, f)) / np.linalg.norm(E.T @ f / np.inner(f, f)))) > tol:
                Xi[:, k] = E.T @ f / np.inner(f, f)
                Xi[:, k] = Xi[:, k] / np.linalg.norm(Xi[:, k])
                t = E @ Xi[:, k]
                Omega[:, k] = F.T @ t / np.inner(t, t)
                f = F @ Omega[:, k] / np.inner(Omega[:, k], Omega[:, k])
            Principale_components[:, k] = E.T @ t / np.inner(t, t)
            E = E - np.outer(t, Principale_components[:, k])
            F = F - np.outer(t, Omega[:, k])
        self.fitted_vars = Xi, Omega, Principale_components, x_centroid, y_centroid

    def predict(self, X):
        Xi, Omega, Principale_components, x_centroid, y_centroid = self.fitted_vars
        m_test, _ = X.shape
        no_of_latent_vars = Omega.shape[1]
        e = X - x_centroid
        intermediate_var = np.zeros((m_test, no_of_latent_vars))
        for k in range(no_of_latent_vars):
            intermediate_var[:, k] = e @ Xi[:, k]
            # np.outer as we compute it for the whole matrix
            e = e - np.outer(intermediate_var[:, k], Principale_components[:, k])
        y_pred = np.c_[y_centroid[0] * np.ones(m_test), y_centroid[1]
                       * np.ones(m_test)] + intermediate_var @ Omega.T
        y = [1 if y_pred[i, 0] > y_pred[i, 1] else 0 for i in range(y_pred.shape[0])]
        return y


class LSIPLS(TransformerMixin, BaseEstimator):
    def __init__(self, no_of_latent_vars=2):
        self.no_of_latent_vars = no_of_latent_vars

    def fit(self, X, y):
        self.Xi = LSIPLS_fit(X, y, self.no_of_latent_vars)
        return self

    def transform(self, X):
        return X @ self.Xi
