install.packages("tidyverse")

library(tidyverse)
library(caret)
library(dplyr)

setwd("D:/sanduni/Level 3/Sem II/ST 3082 - Statistical Learning I/Data set")

flight_data <- read.csv("Clean_Dataset.csv")
flight_data <- flight_data %>% select(-c(X, flight))

# Split the data into training and testing sets 
set.seed(123)
train_indices <- createDataPartition(flight_data$price, p = 0.8, list = FALSE)
train_data <- flight_data[train_indices, ]
test_data <- flight_data[-train_indices, ]
#View(train_data)
#View(test_data)
#dim(train_data)

# Convert categorical variables to factors
train_data <- train_data %>%
  mutate(across(c(airline, source_city, stops, destination_city, class, departure_time, arrival_time), as.factor))
test_data <- test_data %>%
  mutate(across(c(airline, source_city, stops, destination_city, class, departure_time, arrival_time), as.factor))


# One-hot encoding of categorical variables 
dummy_vars_train <- dummyVars(" ~ airline + source_city + stops + destination_city + 
                                class + departure_time + arrival_time", 
                              data = train_data, 
                              fullRank = TRUE)
train_dummy_data <- predict(dummy_vars_train, newdata = train_data)
train_encoded_df <- cbind(train_data[, c("duration", "days_left", "price")], 
                          as.data.frame(train_dummy_data))



dummy_vars_test <- dummyVars(" ~ airline + source_city + stops + destination_city + 
                                class + departure_time + arrival_time", 
                             data = test_data, 
                             fullRank = TRUE)
test_dummy_data <- predict(dummy_vars_test, newdata = test_data)
test_encoded_df <- cbind(test_data[, c("duration", "days_left", "price")], 
                         as.data.frame(test_dummy_data))

head(train_encoded_df)
# Standardization 
numerical_cols <- c("duration", "days_left")
preProc <- preProcess(train_encoded_df[, numerical_cols], method = c("center", "scale"))

train_scaled <- predict(preProc, train_encoded_df)
test_scaled <- predict(preProc, test_encoded_df)


# Combine scaled numerical features with one-hot encoded features
train_data_processed <- cbind(train_scaled[, numerical_cols], 
                              train_encoded_df[, !names(train_encoded_df) %in% numerical_cols])
test_data_processed <- cbind(test_scaled[, numerical_cols], 
                             test_encoded_df[, !names(test_encoded_df) %in% numerical_cols])

# Separate features and target variable
X_train <- train_data_processed[, names(train_data_processed) != "price"]
y_train <- train_data_processed$price
X_test <- test_data_processed[, names(test_data_processed) != "price"]
y_test <- test_data_processed$price

head(train_data_processed)
head(test_data_processed)

# Load the required package for ridge regression
library(glmnet)

# Set the response variable and predictor variables
x=model.matrix(price~.-1,train_data_processed)
fix(x)
y=y_train
# Fitting ridge regression
fit.ridge=glmnet(x,y,alpha=0)
plot(fit.ridge,xvar="lambda",label=TRUE,lw=2)

# Doing cross validation to select the best lambda
cv.ridge=cv.glmnet(x,y,alpha=0)
plot(cv.ridge)    
bestlam =cv.ridge$lambda.min 
bestlam
# Fitting the ridge regression model under the best lambda
#out=glmnet(x,y,alpha=0)
#predict(out ,type="coefficients",s=bestlam)[1:38,]
#coef(out,s=bestlam)
coef(fit.ridge,s=bestlam)

y_pred_train <- predict(fit.ridge, s = bestlam, newx = x)

# Compute Training MSE
train_mse <- mean((y - y_pred_train)^2)
print(paste("Training MSE:", train_mse))

train_rmse = sqrt(train_mse)
print(paste("Training RMSE:", train_rmse))

# Compute Training R^2
ss_res <- sum((y - y_pred_train)^2)
ss_tot <- sum((y - mean(y))^2)
train_r2 <- 1 - (ss_res / ss_tot)
print(paste("Training R^2:", train_r2))

# Set the response variable and predictor variables
x_test=model.matrix(price~.-1,test_data_processed)
fix(x_test)
y_test=y_test
# Use the Ridge model trained on training data to predict test set
y_pred_test <- predict(fit.ridge, s = bestlam, newx = x_test)

# Compute Test MSE
test_mse <- mean((y_test - y_pred_test)^2)
print(paste("Test MSE:", test_mse))

test_rmse = sqrt(test_mse)
print(paste("Test RMSE:", test_rmse))

# Compute Test R^2
ss_res <- sum((y_test - y_pred_test)^2)
ss_tot <- sum((y_test - mean(y_test))^2)
test_r2 <- 1 - (ss_res / ss_tot)
print(paste("Test R^2:", test_r2))

