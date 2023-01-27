'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

class BinDetector():
    def __init__(self):
        """
        This method loads the parameters which were saved during training
        """
        folder_path = os.path.dirname(os.path.abspath(__file__))
        self._mean = np.load(os.path.join(folder_path, 'mean_train_bins.npy'))
        self._var = np.load(os.path.join(folder_path, 'var_train_bins.npy'))
        self._priors = np.load(os.path.join(folder_path, 'priors_train_bins.npy'))

    def _predict(self, x):
        """
        x : 1x3 numpy array to be labelled
        This private method labels each pixel as 1,2,3,4 or 5.
        """
        posteriors = []
        classes = [1, 2, 3]
        for i, c in enumerate(classes):
            prior = np.log(self._priors[i])
            posterior = prior + np.sum(np.log(self.gaussian_pdf(i, x)))
            posteriors.append(posterior)
        return classes[np.argmax(posteriors)]

    def predict(self, X):
        """
        This method labels an nx3 array into corresponding 1,2,3,4 or 5
        (Blue, Non-Blue, Green, Brown,Black)
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

    def segment_image(self, img):
        """
			Obtain a segmented image using Naive Bayes color classifier
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		"""
        #image is converted to YCbCr color space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        X = np.reshape(img, (-1, 3))
        X = np.round(X / 255, decimals=3)
        y_predictions = self.predict(X)
        mask_img = [y  if y==1 else 0 for y in y_predictions]
        return np.reshape(mask_img, (img.shape[0], img.shape[1]))

    def get_bounding_boxes(self, img):
        '''
            Find the bounding boxes of the recycling bins
            Inputs:
                img - segmented image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
        '''
        ################################################################

        img = np.uint8(img)
        #Morphological operations
        # img = cv2.erode(img, (5,5), iterations=12)
        # img = cv2.medianBlur(img, 11)
        img = cv2.erode(img, (5, 5), iterations=2)
        img = cv2.medianBlur(img, 9)
        _, bw_image = cv2.threshold(img, 0,255,cv2.THRESH_BINARY)
        contours_list, hierarchy = cv2.findContours(bw_image,
                                                    cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours
        boxes = []
        for c in contours_list:
            x, y, w, h = cv2.boundingRect(c)
            # Make sure contour area is large enough and height is greater than width
            if w * h > 10000 and h>w:
                boxes.append([x,y,x+w,y+h])
        ################################################################

        return boxes

if __name__ == "__main__":
    folder = 'data/validation'
    for file in os.listdir(folder):
        if file.endswith((".jpg")):
            bd = BinDetector()
            img = cv2.imread(os.path.join(folder, file))
            img1 = img.copy()
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            mask_img = bd.segment_image(img)
            plt.imshow(mask_img)
            plt.show()
            bd.get_bounding_boxes(mask_img)
            boxes =bd.get_bounding_boxes(mask_img)
            print(boxes)
            for x1,y1,x2,y2 in boxes:
                img1 = cv2.rectangle(img1, (x1, y1), (x2, y2), (255, 0, 0), 5)
            plt.imshow(img1)
            plt.show()

#OUTPUT BOXES
# 0061 [[203, 168, 309, 293]]
# 0062 [[29, 374, 133, 499]]
# 0063 [[174, 108, 266, 233]]
# 0064 [[359, 122, 458, 272]]
# 0065 [[814, 431, 928, 623]]
# 0066 []
# 0067 [[584, 319, 695, 506], [709, 318, 824, 509]]
# 0068 []
# 0069 []
# 0070 []
