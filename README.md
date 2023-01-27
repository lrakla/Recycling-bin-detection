# Color and Bin Detection
To run the code please install the following packages
- opencv-python==4.1.2.30
- numpy
- pillow==7.0.0
- scikit-image==0.16.2
- timeout-decorator
- glob2
- sklearn 

### Pixel Classifier
1. Running `test_pixel_classifier.py` will give you the 
for number of blue pixels accurately detected.
Other modules which are used in the process - 
- `generate_rgb_data.py` It has two methods - `read_pixels` and `get_data`
which are used in `train_nb.py`
- `train_nb.py`  - this module fits the RGB data using Naive Bayes and saves mean, variance and priors as .npy files
- `pixel_classifier.py` - this module classifies a given pixel as R,G or B


### Bin Detection
1. Running `test_bin_detector.py` will give an 
accuracy for the 10 validation images as 0% or 100%. Other modules which are used in the process 
- `label_data.py`  Used to label training images. Based of `test_roipoly.py`
- `train_nb_bins.py` Used to fit training data using Naive Bayes and saves mean, variance and priors
- `bin_detector.py` Used to generate segmented images and bounding boxes
- `comparison.py` Uses sklearn Logistic Regression to train data and find bounding boxes


### Results (on validation data)
1. Naive Bayes for blue pixel classification - **~ 97.5%** accuracy
2. Naive Bayes for Bin Detection - **100%** accuracy
3. Multiclass Logistic Regression for BinDetection - **90%** accuracy

### Results (on test/autograder)
1. Naive Bayes for pixel classification - **98%** accuracy
2. Naive Bayes for Bin Detection - **92.5%** accuracy


