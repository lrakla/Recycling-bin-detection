'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

class BinDetector():
    def __init__(self):
        pass
    def segment_image(self, clf, img):
        '''
			Obtain a segmented image using a Logistic Regressioncolor classifier,
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        X = np.reshape(img, (-1, 3))
        X = np.round(X / 255, decimals=3)
        y_predictions = clf.predict(X)
        mask_img = [y if y == 1 else 0 for y in y_predictions]
        return np.reshape(mask_img, (img.shape[0], img.shape[1]))

    def get_bounding_boxes(self, img):
        '''
            Find the bounding boxes of the recycling bins
            Inputs:
                img - original image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE

        # Replace this with your own approach\
        img = np.uint8(img)
        # img = cv2.erode(img, (5,5), iterations=12) #2
        # img = cv2.medianBlur(img, 11) #9
        img = cv2.erode(img, (5, 5), iterations=2)  # 2
        img = cv2.medianBlur(img, 9)  # 9
        _, bw_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        contours_list, hierarchy = cv2.findContours(bw_image,
                                                    cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)

        # Draw first contour
        boxes = []
        for c in contours_list:
            x, y, w, h = cv2.boundingRect(c)
            # Make sure contour area is large enough
            if w * h > 10000 and h > w:
                boxes.append([x, y, x + w, y + h])
        # YOUR CODE BEFORE THIS LINE
        ################################################################

        return boxes



def iou(box1,box2):
  '''
    Computes the intersection over union of two bounding boxes box = [x1,y1,x2,y2]
    where (x1, y1) and (x2, y2) are the top left and bottom right coordinates respectively
  '''
  x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
  x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
  inter_area = max(0, (x2 - x1 + 1)) * max(0, (y2 - y1 + 1))
  union_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1) + (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1) - inter_area
  return inter_area/union_area


def compare_boxes(true_boxes, estm_boxes):
  '''
    Compares the intersection over union of two bounding box lists.
    The iou is computed for every box in true_boxes sequentially with respect to each box in estm_boxes.
    If any estm_box achieves an iou of more than 0.5, then the true box is considered accurately detected.
  '''
  num_true_boxes = len(true_boxes)
  if num_true_boxes == 0:
    return float(len(estm_boxes) == 0)

  accuracy = 0.0
  for box1 in true_boxes:
    for box2 in estm_boxes:
      if iou(box1,box2) >= 0.5:
        accuracy += 1.0
        break
  return accuracy / num_true_boxes

if __name__ == '__main__':
    #This will not run as I have not included the .npy files
    X_green = np.load("labelled_data_green2.npy")
    X_blue = np.load("labelled_data_blue1.npy")
    X_non_blue = np.load("labelled_data_nonblue1.npy")
    X_brown = np.load("labelled_data_brown1.npy")
    X_black = np.load("labelled_data_black.npy")
    y_blue, y_non_blue, y_green,y_brown,y_black= np.full(X_blue.shape[0], 1), np.full(X_non_blue.shape[0], 2), \
                                                 np.full(X_green.shape[0], 3),np.full(X_brown.shape[0], 4),np.full(X_black.shape[0], 5)
    X, y = np.concatenate((X_blue, X_non_blue, X_green,X_brown,X_black)), np.concatenate((y_blue, y_non_blue, y_green,y_brown,y_black))

    LogReg = LogisticRegression(random_state=0,multi_class = 'multinomial').fit(X,y)
    print(LogReg.coef_)
    print(LogReg.intercept_)

folder = "data/validation"
my_detector = BinDetector()
for filename in os.listdir(folder):
    if filename.endswith(".jpg"):
        # read one test image
        img = cv2.imread(os.path.join(folder, filename))

        # load ground truth label
        with open(os.path.join(folder, os.path.splitext(filename)[0] + '.txt'), 'r') as stream:
            true_boxes = yaml.safe_load(stream)

        # show image
        for box in true_boxes:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # convert from BGR (opencv convention) to RGB (everyone's convention)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # segment the image
        mask_img = my_detector.segment_image(LogReg,img)

        # detect recycling bins
        estm_boxes = my_detector.get_bounding_boxes(mask_img)

        # The autograder checks your answers to the functions segment_image() and get_bounding_box()

        # measure accuracy
        accuracy = compare_boxes(true_boxes, estm_boxes)

        print('The accuracy for %s is %f %%.' % (filename, accuracy * 100))
