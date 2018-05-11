######
# Adaboost Classifier
# Student Name: Fan Yang
# Student Unity ID: fyang8
######

# Do not clear your workspace
require(rpart) # for decision stump
require(caret)
require(mlbench)
# set seed to ensure reproducibility
set.seed(100)

# calculate the alpha value using epsilon
# params:
# Input: 
# epsilon: value from calculate_epsilon (or error, line 7 in algorithm 5.7 from Textbook)
# output: alpha value (single value) (from Line 12 in algorithm 5.7 from Textbook)
###
calculate_alpha <- function(epsilon){
  alpha <- log((1-epsilon)/epsilon, exp(1))/2
  return(alpha)
  
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
  
  sum <- 0
  for(j in 1:351){
    if(y_true[j] != y_pred[j]){
      sum <- sum + weights[j]
    }
  }
  
  epsilon <- sum/n_elements
  return(epsilon)
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
  
  weight_temp <- c()
  for(i in 1:n_elements){
    if(y_true[i] == y_pred[i]){
      weight_temp[i] <- old_weights[i]*exp(-alpha)
    }else{
      weight_temp[i] <- old_weights[i]*exp(alpha)
    }
  }
  new_weights <- weight_temp/sum(weight_temp)
  return(new_weights)
  
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

  # k <- 5
  # n_elements <- 351
  # train <- Ionosphere
  alpha_array <- c()
  result_list <- list()
  y_true <- as.factor(train[1:n_elements, 33])
  sample_size <- nrow(train)
  weights <- c()
  for(i in 1:sample_size){
    weights <- append(weights, 1/sample_size)
  }
  wl <- list(weights)
  weight_list <- list()
  weight_list <- append(weight_list, wl)
  
  for(j in 1:k){
    test <- train[sample(nrow(train), n_elements, replace = TRUE, prob = weights), ]
    if(sum(test[,33]) == n_elements){
      for(l in 1:length(y_pred)){
        y_pred[l] <- as.factor(1)
      }
    }else if(sum(test[,33]) == -n_elements){
      for(l in 1:length(y_pred)){
        y_pred[l] <- as.factor(-1)
      }
    }else{
      stump <- rpart(Label ~ ., test,  maxdepth = 1, method = "class")
      y_pred <- predict(stump, train, type = "class")
    }
    # print(y_pred)
    y_pred <- as.factor(y_pred)
    result_list <- append(result_list, list(y_pred))
    table(y_true, y_pred)
    epsilon <- calculate_epsilon(weights, y_true, y_pred, n_elements)
    alpha <- calculate_alpha(epsilon)
    alpha_array <- append(alpha_array, alpha)
    new_weights <- calculate_weights(weights, alpha, y_true, y_pred, n_elements)
    weights <- new_weights
    wl <- list(weights)
    weight_list <- append(weight_list, wl)
  }
  
  result <- c()
  result_temp <- c()
  for(i in 1:351){
    result_temp[i] <-  as.numeric(as.character(unlist(result_list[1])))[i]*alpha_array[1]
  }
  for(i in 2 : k){
    result_temp <- result_temp + as.numeric(as.character(unlist(result_list[i])))*alpha_array[i]
  }
    
  # print(result_temp)
  for(i in 1:n_elements){
    if(result_temp[i] >= 0){
      result[i] <- 1
    }else{
      result[i] <- -1
    }
  }
  
  v <- as.vector(result)
  return(v)
    
}

# Code has already been provided here to preprocess the data and then call the adaboost function
# Implement the functions marked with ### before this line
data(Ionosphere)
Ionosphere <- Ionosphere[,-c(1,2)]
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
