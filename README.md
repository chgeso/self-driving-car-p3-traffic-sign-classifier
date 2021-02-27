# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./screenshots/01-data-set-distribution.png "Visualization"
[image2]: ./screenshots/02-color-image.png "Color Image"
[image3]: ./screenshots/03-grayscale-image.png "Grayscale Image"
[image4]: ./screenshots/04-german-website-pics.png "New Image from German Website"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### README

Here is the link to my [project html output](https://github.com/nggsng/self-driving-car-p3-traffic-sign-classifier/blob/main/Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

First of all, before running this, you can download the pickle data from [this](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip). Because of ths size, you need to make "data" directory and then put the pickle data into the directory.      
I used the numpy library to calculate summary statistics of the traffic signs data set:
   
* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of the testing set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data of train, valid, and test is distributed.
The x-axis represents class id to predict the data through my network model. The y-axis represents the percentage of how many images are distributed in each class id for each train, valid, and test data set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to convert the images to grayscale because, in view of detecting edges, I prefer to widen difference between image array elements' values. This means a grayscaled image is more easier than a color image.

Here is an example of a road work image before and after grayscaling.

|     color  image    |  grayscaling image  |
|:-------------------:|:-------------------:|
| ![alt text][image2] | ![alt text][image3] |

As a last step, I normalized the image data because of helping detecting the solution logits when it is training.  
Normalization make the mean of image elements' values to 0 and equal the variance. It is a well conditional environment to detect the solution logits.

Therefore, the difference between the original data set and the augmented data set is like below.   
* To make wider gaps between image array elements' values by grayscaling.
* To make a good conditional for detecting logits by normalizing.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Dropout	        	|                                				|
| Convolution 3x3	    | 1x1 stride. valid padding, outputs 14x14x128  |
| Max pooling	      	| 2x2 stride,  outputs 7x7x128  				|
| Dropout	        	|                                				|
| Fully connected		| inputs 6272, outputs 3500						|
| Fully connected		| inputs 3500, outputs 200						|
| Fully connected		| inputs 200,  outputs 43						|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an optimizer as AdamOptimizer. The batch size is 128 and the number of epochs is 10. The learning late is 0.001. When I make my network model, the key point is to add dropout to LeNet model sourced from Yann Lecun. According to LeNet model, I couldn't avoid to meet overfitting when I evaluate the valid data set. Therefore, by using dropout method, I could resolve the problem and then I could see a good logits after 10 epochs. In my network model, you can see the better accuracy whenever increasing epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996.
* validation set accuracy of 0.951.
* test set accuracy of 0.926.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? I choose LeNet architecture. because the architecture is easily understandable than others and I could have some configurable points than other architecture.
* What were some problems with the initial architecture? The problem is overfitting to use it originally.
* How was the architecture adjusted and why was it adjusted? To resolve the problem, I used "dropout" after max pooling. 
* Which parameters were tuned? How were they adjusted and why? I have tuned the output depth of convolution, strides, ksize, padding, and outputs o of full connected outputs. The key points for tuning are appropriate size valance for finding coefficients in each fiters. In my case, the sensitive point are in full connection. The output depth of full connected outputs affects the accuracy than others. Therfore, I concentrated how to tune the output size of full connection especially.
* What are some of the important design choices and why were they chosen? By convolutional layers, I could collect information of each parts' importance degree. After RELU, through max pooling, I choose the more important part from given collection. And then I use dropout to reduce overfitting. About fully connection, I need to connect the network model into final outputs which represent class ids. When I connect to just one layer, the performance is not good. Because, the number of the inital inputs is too big (Inputs: 6272). Therefore, I use three layers to connected the network model fully for avoding losing some important information.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]

The first image might be difficult to classify because some parts are bending.   
The second image might be difficult to classify because the brightness is weak.   
The third image might be difficult to classify because the background behind the traffic sign could cause a confusion when detecting the edges.  
The fourth image might be difficult to classify because the brightness is weak.   
The fifth image might be difficult to classify because the background behind the traffic sign is complex.   

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			                           |     Prediction	   	                      | 
|:----------------------------------------:|:----------------------------------------:| 
| Speed limit (30km/h)     		           | Speed limit (30km/h)                     | 
| Keep right     			               | Keep right 			   			      |
| General caution					       | General caution				          |
| Vehicles over 3.5 metric tons prohibited | Vehicles over 3.5 metric tons prohibited |
| Road work		                           | Road work      						  |


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

For the first image, the model is sure that this is a Speed limit (30km/h) (probability of 9.99986649e-01), and the image does contain a Speed limit (30km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99986649e-01      	| Speed limit (30km/h)   						| 
| 8.66053142e-06     	| Speed limit (50km/h)							|
| 2.31313288e-06		| Speed limit (70km/h)							|
| 1.59934746e-06	   	| Speed limit (80km/h)			 				|
| 8.69376038e-07		| Speed limit (20km/h)   						|


For the second image, the model is sure that this is a Keep right (probability of 1.00000000e+00), and the image does contain a Keep right. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00    	| Keep right 									| 
| 2.47890285e-11     	| Go straight or right				     		|
| 8.48541344e-13		| Speed limit (50km/h)							|
| 6.40716646e-13	    | Speed limit (80km/h)	    	 				|
| 5.57683457e-13		| Turn left ahead                   			|

For the third image, the model is sure that this is a General caution (probability of 9.99994159e-01), and the image does contain a General caution. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99994159e-01       	| General caution								| 
| 5.67918369e-06     	| Traffic signals    							|
| 6.66495694e-08		| Pedestrians      			            		|
| 1.66946579e-09	    | Road narrows on the right                 	|
| 2.89160987e-11		| Right-of-way at the next intersection   		|

For the fourth image, the model is sure that this is a Vehicles over 3.5 metric tons prohibited (probability of 1.00000000e+00), and the image does contain a Vehicles over 3.5 metric tons prohibited. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00       	| Vehicles over 3.5 metric tons prohibited		| 
| 1.00000000e+00     	| Speed limit (100km/h)							|
| 1.48322696e-08		| Roundabout mandatory							|
| 1.08836794e-10	   	| No passing for vehicles over 3.5 metric tons	|
| 5.16306858e-11		| Speed limit (80km/h)  						|

For the fifth image, the model is relatively sure that this is a Road work (probability of 1.00000000e+00), and the image does contain a Road work. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00       	| Road work   									| 
| 4.07221634e-09     	| Go straight or right							|
| 4.33405360e-11		| General caution								|
| 2.62549444e-11	    | Bicycles crossing     		 				|
| 1.14924034e-11		| Road narrows on the right 					|
