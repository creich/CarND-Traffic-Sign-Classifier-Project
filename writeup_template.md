# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[sign_hist]: ./images_for_review/sign_histogram.png "sign histogram"
[color_image]: ./images_for_review/color_image.png "colored street sign image"
[gray_image]: ./images_for_review/gray_image.png "grayscale street sign image"
[dark_image]: ./images_for_review/dark_image.png "dark street sign image"
[CLAHE_image]: ./images_for_review/CLAHE_image.png "image preprocessed with CLAHE (Contrast Limited Adaptive Histogram Equalization)"
[noise_image]: ./images_for_review/noise_image.png "image augmented with random noise"
[5_web_samples]: ./images_for_review/5_web_samples.png "5 random samples from the online database"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed between
the different classes

![sign_hist]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I kept data preprocessing pretty simple, yet obviously effective. For this project i only converted the images
to grayscale, applied a 'Contrast Limited Adaptive Histogram Equalization' (CLAHE) and after that normalized the data.
i guess the biggest information gain came with the CLAHE, since there've been a bunch of under- and over-exposed
images within the given data. Normalization somehow felt like the 'standard precedure' :)

Here is an example of a traffic sign image before and after grayscaling.

![color_image]
![gray_image]

Here is an example of a traffic sign image before and after CLAHE.

![dark_image]
![CLAHE_image]


For data augmentation i was trying some things like adding some blur and/or some random noise. In the end i didn't use the blur, since it seemed not to be very helpful and the results have been good already. genrealy i would expect the blur to help in the same way as adding some noise -> prevent overfitting. But again, in this case, adding some noise was much more effective for me than adding some blur.
Another possible step could have been to augment especially the data regarding signs, that came with only a few samples
from the original data set. But since both, the results and the training time were acceptable already and my personal time left for this project was short, i left it the way it is right now.

Maybe in the future i'll have a look and try to improve my results.

Here is an example of an original image and an augmented image:

![noise_image]

The difference between the original data set and the augmented data set is basically that i doubled the amount of training data.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

As proposed, i took LeNet5 as starting point for my network. But i changed a few details. First i changed the input to take images with one channel (for the grayscale images). I also dropped one fully connected layer. So basically i only flattened the last convolutional layer, concatenated it with the intermediate results und fully conected to generate the 43 logits. The idea to connect the intermediate results with the direkt output of the network cam from the paper mentioned within the project description. (-> http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| Input (5x5x16 (from max pooling) + 10x10x16 (intermediate result from second Convolution = 2000), Output 120 |
| RELU					|												|
| Fully connected		| Input (120 (from Fully Connected 1) + 10x10x16 (intermediate result from second Convolution = 2000), Output 43 |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, i basically took the stuff from the LeNet as a starting point. From that i kept almost everything (e.g. the AdamOptimizer). I only changed the hyperparametres, which are the following:

| Hyperparameter         		|     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| learning rate         		| 0.0005   							| 
| EPOCHS | 10 |
| BATCH_SIZE | 64 |


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

during the approach, i noticed that prerpocessing the data is almost more valuable than a good architecture itself. the data available to train the network is the essential key to success. no matter if you directly have it, or even artificially create it. especially if we're talking about images. there are so many well known techniques to enhance the qulity of an image, even for the human eye.

since i used the well known LeNet5 as a basis (as proposed by the course), i already had a very good starting point. combined with good preprocessing, it almost gave the necessary results. but since i read the paper that was linked within the project descripton, i got curious about the direct connection between the first convolutional layer and the fully connected layers. in fact, those changes increased the accurecy and even slightly the robustness of the net.
for whatever reason, i got the impression that adding some dropout layers in variuous points didn't help much. instead it sometimes even decreased the accuracy even on the test data.

so far i found this lesson very insightful and got a lot better feeling of how to apply my new knowledge. but i also realized that it might take a lot more expreience to design a neural net.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![5_web_samples]

I just picked them randomly from the complete German Traffic Sign Dataset. The last two of them are pretty dark. especially the last one is barely visible even for the human eye. Here the CLAHE comes into play, since without it the model would have (as well as a human) had some diffculties to determine the right classification. You can clearly see that even using CLAHE the NET has some difficulties, if you check the given prediction probabilities in the tables in point 3.
The second image is quite the opposite of that,, since it's pretty bright, which also might affect the correct classification. again, using CLAHE relly helps to flatten the histogram, which makes the usefull information stand out more precise.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Vehicles over 3.5 metric tons prohibited      		| Vehicles over 3.5 metric tons prohibited   									| 
| Priority road     			| Priority road 										|
| No passing for vehicles over 3.5 metric tons					| No passing for vehicles over 3.5 metric tons											|
| Speed limit (50km/h)	      		| Speed limit (50km/h)					 				|
| No passing for vehicles over 3.5 metric tons			| No passing for vehicles over 3.5 metric tons      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. I tried it several time, and usually got 4-5 out of 5, which compares pretty well to the given accuracy on the test data set. so i feel the sollution is pretty robust, but would recommend further testing before any real time usage. maybe some data from a completely different source e.g.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

Here are the top three soft max probabilities for each of those 5 images:

correct prediction: 16 -- Vehicles over 3.5 metric tons prohibited

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .45         			| 16 -- Vehicles over 3.5 metric tons prohibited   									| 
| .21     				| 9 --	No passing 										|
| .20					| 10 --	No passing for vehicles over 3.5 metric tons											|
| .17					| 7 --	Speed limit (100km/h)											|
| .17					| 40 --	Roundabout mandatory											|

correct prediction: 12 -- Priority road

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .53         			| 12 --	Priority road				| 
| .23         			| 9 --	No passing				| 
| .12         			| 10 --	No passing for vehicles over 3.5 metric tons				| 
| .11         			| 40 --	Roundabout mandatory				| 
| .07         			| 35 --	Ahead only				| 

correct prediction: 10 -- No passing for vehicles over 3.5 metric tons

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .42         			| 10 --	No passing for vehicles over 3.5 metric tons				| 
| .21         			| 7 --	Speed limit (100km/h)				| 
| .15         			| 5 --	Speed limit (80km/h)				| 
| .07         			| 9 --	No passing				| 
| .04         			| 16 --	Vehicles over 3.5 metric tons prohibited				| 

correct prediction: 2 -- Speed limit (50km/h)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .40         			| 2 --	Speed limit (50km/h)				| 
| .30         			| 5 --	Speed limit (80km/h)				| 
| .26         			| 3 --	Speed limit (60km/h)				| 
| .18         			| 1 --	Speed limit (30km/h)				| 
| .05         			| 4 --	Speed limit (70km/h)				| 

correct prediction: 10 -- No passing for vehicles over 3.5 metric tons

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .24         			| 10 --	No passing for vehicles over 3.5 metric tons				| 
| .09         			| 16 --	Vehicles over 3.5 metric tons prohibited				| 
| .09         			| 7 --	Speed limit (100km/h)				| 
| .06         			| 9 --	No passing				| 
| .06         			| 12 --	Priority road				| 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


