import numpy as np
import pandas as pd

class naive_bayes():
    def fit(self, X, y):
        """
        X:nx3 Array used as training
        y :nx1 Labels used for training
        This method saves the parameters mean, variance and priors in .npy files
        """
        n_samples,n_features = X.shape
        classes = np.unique(y)
        n_classes = len(np.unique(y))
        _mean = np.zeros((n_classes,n_features),dtype = np.float64)
        _var = np.zeros((n_classes,n_features),dtype = np.float64)
        _priors = np.zeros((n_classes), dtype=np.float64)

        for i,c in enumerate(classes):
            X_c = X[y==c]
            _mean[i,:] = X_c.mean(axis = 0)
            _var[i,:] = X_c.var(axis = 0)
            _priors[i] = X_c.shape[0]/n_samples
        np.save('mean_train_bins.npy',_mean)
        np.save('var_train_bins.npy', _var)
        np.save('priors_train_bins.npy', _priors)


if __name__ == '__main__':
    X_green = np.load("labelled_data_green2.npy")
    X_blue = np.load("labelled_data_blue1.npy")
    X_non_blue = np.load("labelled_data_nonblue1.npy")
    X_brown = np.load("labelled_data_brown1.npy")
    X_black = np.load("labelled_data_black.npy")
    #Label 5 classes
    y_blue, y_non_blue, y_green,y_brown,y_black= np.full(X_blue.shape[0], 1), np.full(X_non_blue.shape[0], 2), \
                                                 np.full(X_green.shape[0], 3),np.full(X_brown.shape[0], 4),np.full(X_black.shape[0], 5)
    X, y = np.concatenate((X_blue, X_non_blue, X_green,X_brown,X_black)), np.concatenate((y_blue, y_non_blue, y_green,y_brown,y_black))
    #Fit the model
    clf = naive_bayes()
    clf.fit(X,y)


