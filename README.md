# Machine Vision Project
 Group 23 computing project for ME 5405
 *Due: 20 NOV*
Matlab codes are in "main.m"

## Checklist

### Image 1
[ ]Display the original image on screen. Convert the image to grayscale image. Experiment with contrast enhancement of the image. Comment on the results. \
[x]Straighten the characters “HELLO!” in the image using the three different interpolation methods mentioned in class. Compare and comment on the results. \
[ ]Implement and apply a 5x5 averaging filter to the image. Experiment with filters of different sizes. Compare and comment on the results of the respective smoothing 
methods. \
[ ]Implement and apply a high-pass filter on the image in the frequency domain.  Compare and comment on the resultant image in the spatial domain. \
[ ]Segment the image to separate and label the different characters as clearly as possible. 

### Image 2
[ ] Display the original image on screen.\
[ ] Create an image which is a sub-image of the original image comprising the middle line – HELLO, WORLD.\
[ ] Create a binary image from Step 2 using thresholding. \
[ ] Determine a one-pixel thin image of the characters. \
[ ] Determine the outline(s) of characters of the image.\ 
[ ] Segment the image to separate and label the different characters.\
[ ] Using the training dataset provided on Convas (p_dataset_26.zip), train the (conventional) unsupervised classification method of your choice (i.e., self-ordered maps (SOM), k-nearest neighbors (kNN), or support vector machine (SVM)) to recognize the different characters (“H”, “E”, “L”, “O”, “W”, “R”, “D”). You should use 75% of the dataset to train your classifier, and the remaining 25% for validation (testing). Then, test your trained classifier on each character in Image 2, reporting the final classification results. Do not use the characters in Image 2 as training data for your classifier.\
[ ] Throughout step 7 (training of the classifier), also experiment with pre-processing of the data (e.g., padding/resizing input images) as well as with hyperparameter tuning. In your report, discuss how sensitive your approach is to these changes.\
