from generate_rgb_data import get_data
import numpy as np
import pandas as pd


class naive_bayes():
    def fit(self, X, y):
        """
        X:nx3 Array used as training
        y :nx1 Labels used for training
        This method saves the parameters mean, variance and priors in .npy files
        """
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(np.unique(y))
        _mean = np.zeros((n_classes, n_features), dtype=np.float64)
        _var = np.zeros((n_classes, n_features), dtype=np.float64)
        _priors = np.zeros(n_classes, dtype=np.float64)

        for i, c in enumerate(classes):
            X_c = X[y == c]
            _mean[i, :] = X_c.mean(axis=0)
            _var[i, :] = X_c.var(axis=0)
            _priors[i] = X_c.shape[0] / n_samples
        np.save('mean_train.npy', _mean)
        np.save('var_train.npy', _var)
        np.save('priors_train.npy', _priors)


if __name__ == '__main__':
    train_folder = 'data/training'
    X_train, y_train = get_data(train_folder)
    clf = naive_bayes()
    clf.fit(X_train, y_train)
