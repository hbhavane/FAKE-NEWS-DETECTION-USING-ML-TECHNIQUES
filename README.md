# Project 1: Sentiment Analysis on Movie Reviews

## Overview
Sentiment analysis (also known as opinion mining or emotion AI) refers to the use of natural
language processing, text analysis, computational linguistics, and biometrics to systematically identify,
extract, quantify, and study affective states and subjective information. Sentiment analysis is widely applied
to the voice of the customer materials such as reviews and survey responses. A basic task in sentiment
analysis is classifying the polarity of a given text at the document, sentence, or feature/aspect level —
whether the expressed opinion is positive, negative, or neutral.

## Dataset
The dataset used in this project is the IMDB Dataset that contains 50,000 movie reviews, split
evenly into 25,000 reviews for training and 25,000 reviews for testing. The dataset is labeled with binary
sentiment labels, where positive reviews are labeled as 1 and negative reviews are labeled as 0. The
dataset is available in a CSV file format.

## Visualization

![image](https://github.com/hbhavane/Sentiment-Analysis-on-Movie-Reviews/assets/78750775/2dc08295-a427-4389-a92c-e495b0f0a8fa)  ![image](https://github.com/hbhavane/Sentiment-Analysis-on-Movie-Reviews/assets/78750775/68a27954-273e-42bf-9ce9-dfc79feb2a30)


## Data Modelling

Applying five different machine learning models for sentiment analysis on the IMDB movie reviews
dataset. The models used Ire Logistic Regression, LinearSVC, KNeighborsClassifier, Fully connected layers,
and CNN.

### 1) Logistic Regression:
Logistic Regression is a popular machine learning algorithm used for binary classification problems. It
models the probability of an event occurring, given a set of input features. In this project, I used logistic
regression to classify movie reviews as either positive or negative.
Logistic Regression, I used the following hyperparameters:
Learning rate: 0.01
Number of iterations: 1000

### 2) LinearSVC:
LinearSVC is a linear support vector machine algorithm used for classification problems. It finds the best
separating hyperplane that can separate the classes. In this project, I used LinearSVC to classify movie
reviews as either positive or negative.
LinearSVC, I used the default hyperparameters.

### 3) KNeighborsClassifier:
KNeighborsClassifier is a machine learning algorithm used for classification problems. It is a type of
instance-based learning or non-generalizing learning where the function is only approximated locally and
all computation is deferred until classification. In this project, I used KNeighborsClassifier to classify movie
reviews as either positive or negative.
KNeighborsClassifier, I used the following hyperparameters:
Number of neighbors: 5

### 4) Fully-connected layers:
Fully-connected layers are also known as dense layers. In this type of layer, each neuron in one layer is
connected to every neuron in the previous layer. In this project, I used fully-connected layers for building
a neural network to classify movie reviews as either positive or negative.
For Fully-connected layers, I tried different values for the number of hidden layers and the activation
function. The hyperparameters used for each model are listed below:

![image](https://github.com/hbhavane/Sentiment-Analysis-on-Movie-Reviews/assets/78750775/add1c3f7-dcf9-4517-9c1c-8a59555dc3ee)


### 5) CNN:
CNN stands for Convolutional Neural Networks. They are a type of neural network that is commonly used
for image classification. HoIver, they can also be used for text classification problems. In this project, I used
CNN to classify movie reviews as either positive or negative. I used different numbers of convolutional
layers combined with different numbers of fully-connected layers to compare the results.
For CNN, I tried different combinations of the number of convolutional layers and the number of fully
connected layers. The hyperparameters used for each model are listed below:

![image](https://github.com/hbhavane/Sentiment-Analysis-on-Movie-Reviews/assets/78750775/7717dd35-24c1-48a0-81c7-ba10efdf3931)


I trained and evaluated each model using appropriate evaluation metrics. For all models, I used accuracy
as the evaluation metric.

The results shoId that LinearSVC had the highest accuracy of 88.92%, folloId closely by Logistic Regression
with an accuracy of 88.63%. KNeighborsClassifier had an accuracy of 70.02%, which was loIr than the other
models. Among the Fully-connected layers models, Model 4 with 1 hidden layer and 100 neurons using
sigmoid activation had the highest accuracy of 87.52%. Finally, among the CNN models, Model 3 with 3
convolutional layers (32, 64, and 128 filters) and 3 fully-connected layers (128, 64, and 32 neurons) had
the highest accuracy of 87.54%.
Overall, LinearSVC and Logistic Regression performed the best among the models I tested, folloId by the
Fully-connected layers and CNN models. KNeighborsClassifier had the loIst accuracy among the models.


## Conclusion

### Final comparison is showin in below table and What did I find in Data?

![image](https://github.com/hbhavane/Sentiment-Analysis-on-Movie-Reviews/assets/78750775/cec95c98-8d48-4785-86e3-f08f096f6632)

In conclusion, I built a movie review sentiment analyzer using different machine learning models. I started
by exploring the data and preprocessing it by handling missing values, removing noise and special
characters, transforming all words to loIrcase, tokenizing the words, removing stop words, and stemming
the remaining words. I then split the data into a training set and a test set.
I trained and evaluated five different models: logistic regression, LinearSVC, KNeighborsClassifier, Fully-
connected layers, and CNN. I used appropriate evaluation metrics to compare the performance of each
model.
From our evaluation, I found that the logistic regression model and LinearSVC model both performed Ill,
with accuracy scores of around 88% on the test set. The KNeighborsClassifier model had an accuracy score
of around 73%. The fully-connected layers model and CNN model both performed better than the
KNeighborsClassifier model, with accuracy scores of around 84% and 85%, respectively.
Overall, our results show that different machine learning models can be applied to sentiment analysis
tasks, and their performance can vary depending on the specific problem and dataset. Our sentiment
analyzer can be f






