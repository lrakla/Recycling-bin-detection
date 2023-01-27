'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import os

class PixelClassifier():
  def __init__(self):
    """
    Load the saved parameters of mean, variance and priors.
    """
    folder_path = os.path.dirname(os.path.abspath(__file__))
    self._mean = np.load( os.path.join(folder_path, 'mean_train.npy'))
    self._var = np.load( os.path.join(folder_path, 'var_train.npy'))
    self._priors = np.load( os.path.join(folder_path, 'priors_train.npy'))

  def _predict(self, x):
    """
    x : 1x3 numpy array to be labelled
    This private method labels each pixel as 1,2 or 3.
    """
    posteriors = []
    classes = [1,2,3]
    for i, c in enumerate(classes):
      prior = np.log(self._priors[i])
      posterior = prior + np.sum(np.log(self.gaussian_pdf(i, x)))
      posteriors.append(posterior)
    return classes[np.argmax(posteriors)]

  def predict(self, X):
    """
    This method labels an nx3 array into corresponding 1,2,or 3 (R,G or B)
    Returns y_pred : nx1 array
    """
    y_pred = [self._predict(x) for x in X]
    return np.array(y_pred)

  def gaussian_pdf(self, class_idx, x):
    """
    class_idx : assigns probability to given class_idx
    x : 1x3 array whose probability is to be computed
    """
    mean = self._mean[class_idx]
    var = self._var[class_idx]
    n = np.exp(-((x - mean) ** 2) / (2 * var))
    d = np.sqrt(2 * np.pi * var)
    return n / d

  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE
    y = self.predict(X)
    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y

