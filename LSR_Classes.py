import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# defining the neccessary functions


def LSR_fit(X, y, no_of_latent_vars: int, return_vectors_for_predicting: bool = False):
    E = X
    y2 = -y.copy()
    y2 = y2 + 1
    F = np.c_[y, y2]
    m, n = X.shape
    if return_vectors_for_predicting:
        centroid = 1 / m * X.sum(axis=0)
        y_centroid = 1 / m * F.sum(axis=0)
    xi = np.zeros((n, no_of_latent_vars))
    omega = np.zeros((2, no_of_latent_vars))
    principale_components = np.zeros((n, no_of_latent_vars))
    tol = 1e-10
    for k in range(no_of_latent_vars):
        f = y
        while np.abs(np.linalg.norm(xi[:, k] - (E.T @ f / np.inner(f, f)) / np.linalg.norm(E.T @ f / np.inner(f, f)))) > tol:
            xi[:, k] = E.T @ f / np.inner(f, f)
            xi[:, k] = xi[:, k] / np.linalg.norm(xi[:, k])
            t = E @ xi[:, k]
            omega[:, k] = F.T @ t / np.inner(t, t)
            f = F @ omega[:, k] / np.inner(omega[:, k], omega[:, k])
        principale_components[:, k] = E.T @ t / np.inner(t, t)
        E = E - np.outer(t, principale_components[:, k])
        F = F - np.outer(t, omega[:, k])
    if return_vectors_for_predicting:
        return xi, omega, principale_components, centroid, y_centroid
    else:
        return xi


def LSR_predict(X_test, xi, omega, principale_components, centroid, y_centroid):
    m_test, _ = X_test.shape
    no_of_latent_vars = omega.shape[1]
    e = X_test - centroid
    intermediate_var = np.zeros((m_test, no_of_latent_vars))
    for k in range(no_of_latent_vars):
        intermediate_var[:, k] = e @ xi[:, k]
        # np.outer as we compute it for the whole matrix
        e = e - np.outer(intermediate_var[:, k], principale_components[:, k])
    y_pred = np.c_[y_centroid[0] * np.ones(m_test), y_centroid[1]
                   * np.ones(m_test)] + intermediate_var @ omega.T
    return y_pred


def LSIPLS_fit(X, y, no_of_latent_vars):
    E = X
    f = y - y.mean()
    m, n = X.shape
    W = np.zeros((n, no_of_latent_vars))
    for k in range(no_of_latent_vars):
        W[:, k] = (E.T @ f) / np.linalg.norm(E.T @ f)
        t = E @ W[:, k]
        p = (E.T @ t) / np.inner(t, t)
        q = np.inner(f, t) / np.inner(t, t)
        E = E - np.outer(t, p)
        f = f - q * t
    return W

# defining new Transformers/ the LSR-Classifier


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
        m, n = X.shape
        for i in range(m):
            X[i, :] = X[i, :] / np.linalg.norm(X[i, :], ord=self.norm)
        return X


class LSR_Estimator(BaseEstimator):
    def __init__(self, no_of_latent_vars=2, threshold=0.5):
        self.no_of_latent_vars = no_of_latent_vars
        self.threshold = threshold

    def fit(self, X, y):
        self.xi, self.omega, self.principale_components, self.centroid, self.y_centroid = LSR_fit(
            X, y, self.no_of_latent_vars, return_vectors_for_predicting=True)
        return self

    def predict(self, X):
        ypred = LSR_predict(X, self.xi, self.omega, self.principale_components,
                            self.centroid, self.y_centroid)
        y = [1 if ypred[i, 0] > ypred[i, 1] else 0 for i in range(ypred.shape[0])]
        return y


# brauche BaseEstimator damit get_params und set_params implementiert ist
class LSR_Transformer(TransformerMixin, BaseEstimator):
    def __init__(self, no_of_latent_vars=2):
        self.no_of_latent_vars = no_of_latent_vars

    def fit(self, X, y):
        self.xi = LSR_fit(X, y, self.no_of_latent_vars)
        return self

    def transform(self, X):
        return X @ self.xi


class LSIPLS(TransformerMixin, BaseEstimator):
    def __init__(self, no_of_latent_vars=2):
        self.no_of_latent_vars = no_of_latent_vars

    def fit(self, X, y):
        self.W = LSIPLS_fit(X, y, self.no_of_latent_vars)
        return self

    def transform(self, X):
        return X @ self.W
