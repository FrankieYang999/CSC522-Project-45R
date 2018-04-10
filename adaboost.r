######
# Adaboost Classifier
# Student Name:
# Student Unity ID:
######

# Do not clear your workspace
library(KernelKnn)
require(rpart) # for decision stump
require(caret)

# set seed to ensure reproducibility
set.seed(100)

# calculate the alpha value using epsilon
# params:
# Input: 
# epsilon: value from calculate_epsilon (or error, line 7 in algorithm 5.7 from Textbook)
# output: alpha value (single value) (from Line 12 in algorithm 5.7 from Textbook)
###
calculate_alpha <- function(epsilon){
  
}

# calculate the epsilon value  
# input:
# weights: weights generated at the end of the previous iteration
# y_true: actual labels (ground truth)
# y_pred: predicted labels (from your newly generated decision stump)
# n_elements: number of elements in y_true or y_pred
# output:
# just the epsilon or error value (line 7 in algorithm 5.7 from Textbook)
###
calculate_epsilon <- function(weights, y_true, y_pred, n_elements){

}


# Calculate the weights using equation 5.69 from the textbook 
# Input:
# old_weights: weights from previous iteration
# alpha: current alpha value from Line 12 in algorithm 5.7 in the textbook
# y_true: actual class labels
# y_pred: predicted class labels
# n_elements: number of values in y_true or y_pred
# Output:
# a vector of size n_elements containing updated weights
###
calculate_weights <- function(old_weights, alpha, y_true, y_pred, n_elements){
  
}

# implement myadaboost - simple adaboost classification
# use the 'rpart' method from 'rpart' package to create a decision stump 
# Think about what parameters you need to set in the rpart method so that it generates only a decision stump, not a decision tree
# Input: 
# train: training dataset (attributes + class label)
# k: number of iterations of adaboost
# n_elements: number of elements in 'train'
# Output:
# a vector of predicted values for 'train' after all the iterations of adaboost are completed
###
myadaboost <- function(train, k, n_elements){

}


# Code has already been provided here to preprocess the data and then call the adaboost function
# Implement the functions marked with ### before this line
data(ionosphere)
Ionosphere <- ionosphere[,-c(1,2)]
# lets convert the class labels into format we are familiar with in class
# -1 for bad, 1 for good (create a column named 'Label' which will serve as class variable)
Ionosphere$Label[Ionosphere$Class == "good"] = 1
Ionosphere$Label[Ionosphere$Class == "bad"] = -1
# remove unnecessary columns
Ionosphere <- Ionosphere[,-(ncol(Ionosphere)-1)]
# class variable
cl <- Ionosphere$Label
# train and predict on training data using adaboost
predictions <- myadaboost(Ionosphere, 5, nrow(Ionosphere))
# generate confusion matrix
print(table(cl, predictions))
