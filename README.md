Here you will learn the essential foundations of Artificial Intelligence (AI) necessary to build and train a neural network.

## Topics Covered:
- Module 1: Introduction to AI Programming with Python
- Module 2: Intro to Python
  - Lessons (Why Python Programming, Data Types and Operators, Control Flow, Functions and Scripting)
  - Project_1 (using an Image classifier)
- Module 3: Numpy, Pandas, Matplotlib
  - Anaconda, Jupyter Notebooks
  - Numpy, Pandas, Matplotib
- Module 4: Linear Algebra Essentials
  - Lessons (Introduction, Vectors, Linear Combination, Linear Transformation and Matrices and Linear Algebra in Neural Networks )
  - Labs (Vectors, Linear Combination and Linear Mapping)
- Module 5: Neural Networks
  - Lessons (Introduction to Neural Networks, Implementing Gradient Descent and Training Neural Networks)
  - Lesson (Deep Learning with PyTorch)
- Module 6: Image Classifier Project


## Numpy Mini Project

### Tasks: 
<details>
  <summary>Mean Normalization</summary>
  <p>In machine learning we use large amounts of data to train our models. Some machine learning algorithms may require that the data is normalized in order to work correctly. The idea of normalization, also known as feature scaling, is to ensure that all the data is on a similar scale, i.e. that all the data takes on a similar range of values. For example, we might have a dataset that has values between 0 and 5,000. By normalizing the data we can make the range of values be between 0 and 1.
  </p>
  <p>
  In this lab, you will be performing a different kind of feature scaling known as mean normalization. Mean normalization will scale the data, but instead of making the values be between 0 and 1, it will distribute the values evenly in some small interval around zero. For example, if we have a dataset that has values between 0 and 5,000, after mean normalization the range of values will be distributed in some small range around 0, for example between -3 to 3. Because the range of values are distributed evenly around zero, this guarantees that the average (mean) of all elements will be zero. Therefore, when you perform mean normalization your data will not only be scaled but it will also have an average of zero.
  </p>
</details>

<details>
  <summary>Data Separation</summary>
  
  <p>After the data has been mean normalized, it is customary in machine learnig to split our dataset into three sets:</p>
   <p> 
    <ul>
      <li>A Training Set</li>
      <li>A Cross Validation Set</li>
      <li>A Test Set</li>
    </ul>
   </p>
    <p>The dataset is usually divided such that the Training Set contains 60% of the data, the Cross Validation Set contains 20% of the data, and the Test Set contains 20% of the data.
  </p>
</details>


## Pandas Mini Project

### Tasks: 
<details>
  <summary>Statistics from Stock Data</summary>
  <p>In this lab we will load stock data into a Pandas Dataframe and calculate some statistics on it. We will be working with stock data from Google, Apple, and Amazon. All the stock data was downloaded from yahoo finance in CSV format. In your workspace you should have a file named GOOG.csv containing the Google stock data, a file named AAPL.csv containing the Apple stock data, and a file named AMZN.csv containing the Amazon stock data. (You can see the workspace folder by clicking on the Jupyter logo in the upper left corner of the workspace.) All the files contain 7 columns of data:
  </p>
</details>


## Project I

<details>
  <summary>Project Summary</summary>
  <h3>Image Classification for a City Dog Show</h3>
  <p>In this project you will use a created image classifier to identify dog breeds. We ask you to focus on Python and not on the actual classifier (We will focus on building a classifier ourselves later in the program).</p>

  <h3>Description:</h3>.
  <p>Your city is hosting a citywide dog show and you have volunteered to help the organizing committee with contestant registration. Every participant that registers must   submit an image of their dog along with biographical information about their dog. The registration system tags the images based upon the biographical information.
  Some people are planning on registering pets that aren’t actual dogs. You need to use an already developed Python classifier to make sure the participants are dogs.
  </p>

  <h3>Tasks:</h3>
  <ul>
    <li>Using your Python skills, you will determine which image classification algorithm works the "best" on classifying images as "dogs" or "not dogs".</li>
    <li>Determine how well the "best" classification algorithm works on correctly identifying a dog's breed.</li>
    <li>If you are confused by the term image classifier look at it simply as a tool that has an input and an output. The Input is an image. The output determines what the image depicts. (for example: a dog). Be mindful of the fact that image classifiers do not always categorize the images correctly. (We will get to all those details much later on the program).
    </li>
    <li>Time how long each algorithm takes to solve the classification problem. With computational tasks, there is often a trade-off between accuracy and runtime. The more accurate an algorithm, the higher the likelihood that it will take more time to run and use more computational resources to run.
    </li>
  </ul>
  
  <h3>Important Notes:</h3>
  <p>For this image classification task you will be using an image classification application using a deep learning model called a convolutional neural network (often abbreviated as CNN). CNNs work particularly well for detecting features in images like colors, textures, and edges; then using these features to identify objects in the images. You'll use a CNN that has already learned the features from a giant dataset of 1.2 million images called ImageNet. There are different types of CNNs that have different structures (architectures) that work better or worse depending on your criteria. With this project you'll explore the three different architectures (AlexNet, VGG, and ResNet) and determine which is best for your application.</p>  
</details>

## Project II

### Tasks: 
<details>
  <summary>Part 1 - Developing an Image Classifier with Deep Learning</summary>
  <p>
In this first part of the project, you'll work through a Jupyter notebook to implement an image classifier with PyTorch.
  </p>
</details>

<details>
  <summary>Part 2 - Building the command line application</summary>
  <p>
Now that you've built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use. Your application should be a pair of Python scripts that run from the command line. For testing, you should use the checkpoint you saved in the first part.
  </p>
</details>














