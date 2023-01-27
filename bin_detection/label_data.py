import numpy as np
import os, cv2
from roipoly import RoiPoly
from matplotlib import pyplot as plt


def label_bins(folder):
    """
    Given a folder, this function returns a nx3 numpy array of labels. The 3 channels are Y,Cr,Cb
    which are normalized between 0 and 1
    """
    # number of files
    X = np.zeros([1, 3])
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # display the image and use roipoly for labeling
        fig, ax = plt.subplots()
        ax.imshow(img)
        my_roi = RoiPoly(fig=fig, ax=ax, color='r')
        # get the image mask
        mask = my_roi.get_mask(img)
        # display the labeled region and the image mask
        temp = np.round(img[mask, :] / 255, decimals=3)
        X = np.vstack((X, temp))
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('%d pixels selected\n' % img[mask, :].shape[0])
        ax1.imshow(img)
        ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
        ax2.imshow(mask)
        plt.show(block=True)
    return X[1:]


if __name__ == '__main__':
    folder = 'data/training'
    X = label_bins(folder)
    # np.save('labelled_data_black.npy', X)

    # used to iteratively add new data to existing .npy files
    with open('labelled_data_blue1.npy', 'rb') as f:
        X_positive = np.load(f)
        X = np.vstack((X_positive, X))
        np.save('labelled_data_blue1.npy', X)
