# We will use the training data to build a classifier using 2 machine learning
# Models (Logistic Regression and Random Forest). The accuracy of the 2 models 
# to be calculated and compared. So, the reliable classifier can be used for 
# future prediction.

# Load Libraries

library('ggplot2') # Data Visualization
library('randomForest') # Random Forest 

# 1- Loading Data
# Load data to the Global Environment

train<- read.csv ("D:/Machine Learning/Kreditech/training.csv",sep=";")

str(train)   # Show Training Dataset Structure and Data Types
view (train) # Show the data

# 2- Data Preparation and Exploration
# Convert variables with factor datatype to numericto be able to calculate 
# correlation value

for (i in 1:ncol(train)) {
  train [,i] = as.numeric(train [,i], na.rm=TRUE)
}

# Calculate the correlation between the classlabel and other features; While 
# prioritizing features with few factor levels

cor(train$v76, train$classlabel)
cor(train$v68, train$classlabel)
cor(train$v7, train$classlabel,use="pairwise.complete.obs")  # Omit NA values
cor(train$v70, train$classlabel,use="pairwise.complete.obs") # Omit NA values
cor(train$v55, train$classlabel,use="pairwise.complete.obs") # Omit NA values
cor(train$v53, train$classlabel,use="pairwise.complete.obs") # Omit NA values
cor(train$v32, train$classlabel,use="pairwise.complete.obs") # Omit NA values
cor(train$v84, train$classlabel,use="pairwise.complete.obs") # Omit NA values
cor(train$v44, train$classlabel,use="pairwise.complete.obs") # Omit NA values

# Confirm my finding

tapply(train$classlabel, train$v68, mean)

# 3- Data Exploration Graphs: for V68 (correlation = 0.9863213) with most significant
# cross-correlation to the classifier
# the 2nd variable in  cross-correlation to the classifier is V95 (correlation
# =  0.2693044) (While the correlation is not very signficant, including it may 
# help; While V68 will remain the dominant dependent variable) 

d <- data.frame(V68 = train$v68[1:3700], label = train$classlabel)
ggplot(d, aes(V68,fill = factor(label))) + geom_histogram()

s <- data.frame(V95 = train$v95[1:3700], label = train$classlabel)
ggplot(s, aes(V95,fill = factor(label))) + geom_histogram()

## 4- Feature Engineering: preparing features used for training and prediction; 
# by choosing features that have significant effect on classlabel
# according to the exploratory process

new_train <- data.frame (dom= train$v68, com= train$v95, label= train$classlabel)
View(new_train)

## 5- Model Training: Using machine learning models to train our new data frame

## (5.1) Logistic Regression Model:
# Replace NA values in Complementary variable (v95) with sample values
for (i in 1:3700) {
    if(is.na(new_train$com[i])){
          new_train$com[i] = sample(na.omit(new_train$com),1)
    }
}
  
set.seed(123)
fit_logit<- glm(label ~ dom + com,data = new_train)

# Predicted Values
fit_logit$fitted.values <- predict(fit_logit)

# Create an empty vector
ans_logit = rep(NA,3700)

# Round the predicted values

for(i in 1:3700) {
  ans_logit[i] = round(fit_logit$fitted.values[i],0)
  }

# Compare predicted values with the Classlabel original values
table(ans_logit)
table(train$classlabel)

# Classifier Evaluation: calculate the accuracy of Logistic Regression Model

a = sum(ans_logit ==1 & train$classlabel == 1)
b = sum(ans_logit ==2 & train$classlabel == 2)
acc = (a + b) / 3700 # Accuracy = 0.9981

## (5.2) Random Forest Regression Model:

set.seed(123)
fit_rf<- randomForest(label ~ dom + com,data = new_train)
rf_fitted <- predict(fit_rf)
ans_rf = rep(NA,3700)

# Round the predicted values

for(i in 1:3700) {
  ans_rf[i] = round(rf_fitted[i],0)
}

# Compare predicted values with the Classlabel original values
table(ans_rf)
table(train$classlabel)

# Classifier Evaluation: calculate the accuracy of Logistic Regression Model

c = sum(ans_rf ==1 & train$classlabel == 1)
d = sum(ans_rf ==2 & train$classlabel == 2)
acc2 = (c + d) / 3700 # Accuracy = 0.9981

# construct Validation data frame