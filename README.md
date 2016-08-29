# Kaggle

A few python scripts that perform well on Kaggle competions.

# Competition
## MNIST

Contains various scripts using either XGBoost, PCA + SVM or Convolutional Neural Networks. 

- CNNs tend to perform the best, with the VGG-like net performing the best but requiring the largest amount of time. 
- SqueezeNet is fast to train but does not perform as well as VGG
- The current best script I posses is the DCCNN MNIST architecture, but since it has been trained on the entire data set it will 
over fit in this Kaggle dataset and give 100% accuracy. (Original accuracy = 99.77 %)

## Titanic

Starter dataset to teach about the basics of Machine Learning principles such as data cleaning / preprocessing and feature construction.

- Best performing model is XGBoost 

## Bike Sharing Demand

A time series data set which is very useful to understand how to manipulate and train time series datasets. 

- Best performing model is an ensemble of two XGBoost which learn the two different time factored outputs and then merge them into a single output.

## BNP

Bank BNP Paribas contest.

- Tried various combinations of stacking and neural nets. Best was a combination of neural nets and XGBoost stacked with Logistic Regression as final layer.
