import numpy as np


class PCA:
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.components_ = None

    def fit(self, X, eta=1e-2, n_iters=1e4):
        def demean(X):
            return X - np.mean(X, axis=0)

        def f(X, w):
            return np.sum(X.dot(w) ** 2) / len(X)

        def df(X, w):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(X, initial_w, eta=1e-2, n_iters=1e4, epsilon=1e-8):
            w = direction(initial_w)
            iter_cnt = 0
            while iter_cnt < n_iters:
                last_w = w
                gradient = df(X, w)
                w = w + eta * gradient
                w = direction(w)
                if (abs(f(X, w) - f(X, last_w)) < epsilon):
                    break
                iter_cnt += 1
            return w

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X_pca.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, initial_w, eta, n_iters)
            self.components_[i] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self

    def transform(self, X):
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        return X.dot(self.components_)

    def __repr__(self) -> str:
        return "PCA(n_components = %d)" % self.n_components
