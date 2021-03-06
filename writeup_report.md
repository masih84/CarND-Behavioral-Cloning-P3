# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

In the fourth project of Udacity Autonomous nano programs, Behavioral Cloning, I practiced training and implementation of the convolutional neural network with [NVIDIA](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) structure using [Keras](https://keras.io/) to train a car driving in the center of the road when it travles around different roads in  self driving car [simulator](https://github.com/udacity/self-driving-car-sim) in [Unity](https://unity3d.com/).

![Sample Autonomous Driving Results](https://raw.githubusercontent.com/masih84/CarND-Behavioral-Cloning-P3/master/track2.gif)

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

A Python code was developed on github [Full Project Repo](https://github.com/masih84/CarND-Behavioral-Cloning-P3) - [model.py](https://github.com/masih84/CarND-Behavioral-Cloning-P3/blob/master/model.py). The project is written in python and its utilises such as [numpy](http://www.numpy.org/) and [OpenCV](http://opencv.org/), [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/).

[//]: # (Image References)

[image1]: ./Network_Structure_mod.png "Model Structure"
[image2]: ./sample_images/Center.png "Center Camera"
[image3]: ./sample_images/Left2.png "Recovery Image"
[image4]: ./sample_images/Center2.png "Recovery Image"
[image5]: ./sample_images/Right2.png "Recovery Image"
[image6]: ./sample_images/Center2.png "Normal Image"
[image7]: ./sample_images/Center-flipped.png "Flipped Image"
[image8]: ./sample_images/track1.png "Track 1 Image"
[image9]: ./sample_images/track_2.png "Track 2 Image"
[image10]: ./sample_images/model_mse.PNG "model MSE Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/masih84/CarND-Behavioral-Cloning-P3/blob/master/model.py). containing the script to create and train the model
* [drive.py](https://github.com/masih84/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/masih84/CarND-Behavioral-Cloning-P3/blob/master/model.h5) containing a trained convolution neural network 
* [writeup_report.md](https://github.com/masih84/CarND-Behavioral-Cloning-P3/edit/master/writeup_report.md) summarizing the results
* [Track1.mp4](https://raw.githubusercontent.com/masih84/CarND-Behavioral-Cloning-P3/master/Track1.mp4) self-driving result Track 1
* [Track2.mp4](https://raw.githubusercontent.com/masih84/CarND-Behavioral-Cloning-P3/master/Track2.mp4) self-driving result Track 2


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I implemented the Neural Network structure used in [NVIDIA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) for driving an actual car. This network architecture is shown in Figure 1 consists of 9 layers, including a normalization and cropping layer, 5 convolutional layers and 3 fully connected layers. The input image is normalized and cropped and passed to the network. The model includes RELU layers to introduce nonlinearity (code lines 120 to 136), and the data is normalized in the model using a Keras lambda layer (code line 113).


#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 74-90). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 156).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, driving multiple laps, driving in opposite directions, flipping the image to be used as another data set, driving in both Track 1 and 2. For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to be able to train the car such that it can drive in both tracks without going off track.

My first step was to use a simple one layer neural network model to check if the pip-line works and I can save the model and run it in the simulator.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that include cropping and normalization. It reduced the error but still, the car could not finish the first turn.

Then I used [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) structure to train the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I updated the model to [NVIDIA](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) with 9 layers.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 108-152) consisted of nine-layesr neural network with the following layers and layer sizes:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Nomalization         		| 160x320x3   							| 
| Cropping         		| 65x320x3    							| 
| Convolution 5x5     	| 2x2 stride, valid padding, activation=RELU, outputs 31x158x24 	|
| Convolution 5x5     	| 2x2 stride, valid padding, activation=RELU, outputs 14x77x36 	|
| Convolution 5x5     	| 2x2 stride, valid padding, activation=RELU, outputs 10x73x48 	|
| Convolution 3x3     	| 1x1 stride, valid padding, activation=RELU, outputs 8x71x64 	|
| Convolution 3x3     	| 1x1 stride, valid padding, activation=RELU, outputs 6x69x64 	|
| Flatten	      	| outputs 26496 				|
| Fully connected	1	| outputs 100  		        									|
| Fully connected	2	| outputs 50  		        									|
| Fully connected	2	| outputs 1  		        									|
|	reduce_mean					|			mse									|
|	optimizer					|				AdamOptimizer								|


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn using more training data. I laso correct streering angle for right and left with 0.2 deg. These images show what a recovery looks like starting from Left, Center and Right :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would increase number of training data. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 85632 number of data points. I used Generators to be able to work with this large amounts of data. Instead of storing the preprocessed data in memory all at once, using a generator I could pull pieces of the data and process them on the fly only when I need them, which is much more memory-efficient.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by trail and error. I used an Adam optimizer so that manually training the learning rate wasn't necessary. Here is the plot of Model Mean Square Error for both Training and validation sets. Since the loss was decreasing for both validation and training sets, the model was not over-fitted.

![alt text][image10]

## Result Video
The trained Model is successfully driving around both track 1 and 2 without going out os the track. I increased the car speed in [drive.py] and CNN control the steering angle without any issues up to 20 MPH on both tracks. I used [video.py](https://github.com/masih84/CarND-Behavioral-Cloning-P3/blob/master/video.py) to generate the project video for both tracks. The [Track one](https://raw.githubusercontent.com/masih84/CarND-Behavioral-Cloning-P3/master/Track1.mp4), and [Track two](https://raw.githubusercontent.com/masih84/CarND-Behavioral-Cloning-P3/master/Track2.mp4) are captured car motion in Autonomous mode from the center camera. I noticed saving the image with video.py slow down my GPU/CPU and affect the car smooth motion. So, I recorded the simulation using [ezvid](https://www.ezvid.com/) and could drive as fast as 20 MPH on  both tracks. Here are Youtube Links: 
* [YouTube Link](https://youtu.be/G8SF40JCmks) Autonomous driving results Track one
* [YouTube Link](https://youtu.be/RC2NaYsrmMM) Autonomous driving results Track two

![alt text][image8]

![alt text][image9]

## Discussion
This project was a great practice and very exciting project to actually implement what we learned Deep learning for the self-driving car. The training was very important and having multiple laps for training and drive in both direction allows the network to learn how to steer in both tracks. Learning the importance of Tenseflow with GPU and using the generator to use limited memory was other handy techniques I learned in this project.

### Problems/Issues faced
The step by step instruction in the project descriptions was very helpful to learn how to do this project. The only challenge I faced was regarding installation of TensorFlow GPU on my Laptop. I followed this [instructions](https://www.quantinsti.com/blog/install-tensorflow-gpu) to install Tenserflow with Cuda version 10 but it did not work and I had this [DLL load failed] error. After some more research, I found from [here](https://github.com/tensorflow/tensorflow/issues/22794) that I should install Cuda 9.0 instead of Cuda 10 with tensorflow1.12.0 and cudnn 7.4.1.5.

