# Semantic Segmentation
### Goal
The goal of this project is to train a fully convolutional Neural network based on VGG-16 classifier to perform semantic segmentation of road images. The ultimate goal is to correctly identify drivable road of an image from a dash cam. I use KITTI data set to train and test the neural network. 

# Project Aproach 

### Network Architecture

Here, I use the pre-trained VGG-16 image classifier and converting it to a fully convolutional network by adding 1x1 convolutions, skip connections and upsampling to the VGG layers 3, 4 and 7. I also set the number of classes to 2 representin g Road and Non-Road areas of the image. To improve the performance, I used skip connections, performing 1x1 convolutions on previous VGG layers 3 and 4 and adding them element-wise to upsampled lower-level layers through transposed convolution. Each convolution and transpose convolution layer includes a kernel initializer and regularizer 

### Loss Function and Optimization

I use a Cross entropy softmax loss function with an Adam optimizer.

### Hyperparameters

keep_prob: 0.5
learning_rate: 0.0009
epochs = 50
batch_size = 10


### Results

Throughout the run, the model decreased the loss gradually starting from 3.782 in the first batch of the 1st epoch to 0.026 in average for the last epoch.
The model was able to successfully classify the road areas from the rest in most of the cases althought there is a lot of room to improve. Below are few sample images.

[//]: # (Image References)

[image1]: ./Samples/1.png "1"
[image2]: ./Samples/2.png "2"
[image3]: ./Samples/3.png "3"
[image4]: ./Samples/4.png "4"
[image5]: ./Samples/5.png "5"
[image6]: ./Samples/6.png "6"
[image7]: ./Samples/7.png "7"
[image8]: ./Samples/8.png "8"

![alt text][image1]  
![alt text][image2]  
![alt text][image3]  
![alt text][image4]  
![alt text][image5]  
![alt text][image6]  
![alt text][image7]  
![alt text][image8]  

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
