# MNIST-Digit-Recognizer
Tensorflow implemented solution for Convolutional Neural Network(CNN) based digit recognition on MNIST dataset from Kaggle competition
(achieve accuracy: 98+ %)

**Competition Description:**

MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.

URL: https://www.kaggle.com/c/digit-recognizer

**Solution:**

DigitRecognizerMNIST.py : Python code for implementating solution by utilizing CNN Architcture(Convolutional Neural Network). Input 28x28 and output is 10 classes to classify digits. 

Below are the layer details:

1) Input layer - 28x28
2) First Convo Layer - input 28x28, kernal 6x6, output 6, stride 1
3) Second Convo Layer - input 6, kernal 5x5, output 12, stride 2
4) Third Convo Layer - input 12, kernal 4x4, output 24, stride 3
5) Fourth FC Layer - input 24 (after flatening), output 200
6) Output Layer - input 200, output 10 (10 classes - softmax)

Learning is being perofrmed using AdamOptimizer.

**Execution Environment & Required Library:**

  1. Python 2.x
  2. Libraries
     - Numpy
     - Pandas
     - Tensorflow
    
**Note for other Enthusiastic Contributor:**

Most welcome to improvise the experiment with better better CNN Architctures.

**Queries?...Connect with me at:**

1) LinkedIn: https://linkedin.com/in/prakash-chandra-chhipa
2) Email: prakash.chandra.chhipa@gmail.com



