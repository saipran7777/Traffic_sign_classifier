# **Traffic Sign Recognition Project** 

## Report

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./custom/gray.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./custom/30.jpg "Speed limit 30"
[image5]: ./custom/70.jpg "Speed limit 70"
[image6]: ./custom/ped.jpg "Pedestrian"
[image7]: ./custom/rt.jpg "Right Turn"
[image8]: ./custom/stop.jpg "STOP"
grayscale
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/saipran7777/Traffic_sign_classifier)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
I used the in-built len() function to find the size of training, validation and test sets. I used .shape to extract the image shape. I used the pandas library to import csv file and find unique labels.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
In exploratory visuzlization of dataset, I used pandas to get the frequency of each label. The highest and lowest frequency were '2010' and '180' corresponding to the labels '2' and ('19','37','0'). Taking a random index, I have also checked if the mapping between X_train and y_train is correct and that their lengths are equal

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it improves computational efficiency by reducing the number of weights required to be computed and readjusted in every EPOCH.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it helps the code run faster and the values to lie between -1 and 1. I decided not to generate additional data because the validation accuracy was greater than .93 . 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is an adaptaion of Le Net model with removal of a max-pool layer. It consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 24x24x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 12x12x16 				|
| Flatten  | Input = 12x12x16. Output = 2304.       |
| Fully connected		| Input = 2304. Output = 120        									|
| Fully connected		| Input = 120. Output = 84      									|
| Fully connected		| Input = 84. Output = 43        									|
| Logits			|         									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I computed cross entropy of the logits and used Adam Optimizer to decrease the cross entropy. I used the default batch size and learning rate of 128 and 0.001 respectively. I changed the EPOCH to 15 as the model was learning till 10 EPOCHS (maybe due to low learning rate)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100 %
* validation set accuracy of 93.4 % 
* test set accuracy of 91.9 %


If a well known architecture was chosen:
* What architecture was chosen?
 LeNet architecture was used 
* Why did you believe it would be relevant to the traffic sign application?
LeNet architecture was used earlier to identify Digits in MNIST dataset. The architecture to identify shapes , patterns decently. Identifying a Traffic sign would also need the model to identify the shapes and patterns . Therefore I believed LeNet would be a great start. Although a small modification in the architecture had to be made to improve accuracy. I removed the first Max pooling layer as I felt that most of the information about the image is getting lost due to it. Removing the Max pool layer improved accuracy from 88.89 % to 93.4%
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 The accuracy of the training and validation test sets were almost the same, which provides an evidence that the model is well trained and works for untrained scenarios.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first two images might be difficult to classify because they contain digits inside them which need to identified correctly. The number of pixels for the digits are less making it further difficult to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 30    		| Speed limit 30 									| 
| Speed limit 70     			| Speed limit 70										|
| Pedestrain | Pedestrain|
| Right Turn Ahead | Right Turn Ahead|
| STOP			| STOP										|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 91.9%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 30    		| 1						| 
| Speed limit 70     			| 0.63								|
| Pedestrain | 0.999|
| Right Turn Ahead | 0.997|
| STOP			| 0.84									|




For the first image, the model is highly sure that this is a Speed limit 30 Sign (probability of 1), and the image does contain a Speed limit 30 Sign. The top five soft max probabilities were [  1.00000000e+00,   9.45942970e-17,   1.06105948e-17, 5.07008272e-20,   4.94053654e-21]

For the second image , the model is relatively sure that it is a Speed limit 70 Sign. The top 5 softmax probabilities were [6.35294735e-01,   3.64679933e-01,   2.54013794e-05, 1.18109176e-08,   6.93225642e-13] corresponding to the indices of [ 4,  1,  0, 39, 37]. This implies that the model was confused in digits classification as the top 3 were having more probability and the corresponding labels were Speed limit (70km/h),Speed limit (30km/h), Speed limit (20km/h) respectively


