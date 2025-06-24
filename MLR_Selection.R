library(leaps)
library(dplyr)
library(caret)

setwd("D:/sanduni/Level 3/Sem II/ST 3082 - Statistical Learning I/Data set")

flight_data <- read.csv("Clean_Dataset.csv")
flight_data <- flight_data %>% select(-c(X, flight))

# Split the data into training and testing sets 
set.seed(123)
train_indices <- createDataPartition(flight_data$price, p = 0.8, list = FALSE)
train_data <- flight_data[train_indices, ]
test_data <- flight_data[-train_indices, ]

# Convert all character columns to factors
train_data[sapply(train_data, is.character)] <- lapply(train_data[sapply(train_data, is.character)], factor)

################################################################################
#            Best Subset Selection                                             #
################################################################################
regfit.full <- regsubsets(price ~ ., data = train_data, nvmax = 30)
regfull.summary <- summary(regfit.full)

plot(regfull.summary$cp, xlab = "Number of Variables", ylab = "Cp")
which.min(regfull.summary$cp)
points(27, regfull.summary$cp[27], pch = 20, col = "red")

plot(regfull.summary$adjr2, xlab = "Number of Variables", ylab = "adjr2")
which.max(regfull.summary$adjr2)
points(29, regfull.summary$adjr2[29], pch = 20, col = "red")

plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "adjr2")

# Custom prediction function for regsubsets
predict.regsubsets = function(object, newdata, id, ...) {
  form = as.formula(object$call[[2]])
  mat = model.matrix(form, newdata)
  coefi = coef(object, id = id)
  mat[, names(coefi)] %*% coefi
}

# Train-validation split
set.seed(1)
train = sample(seq(nrow(train_data)), size = floor(0.8 * nrow(train_data)), replace = FALSE)

regfit.fwd2 = regsubsets(price ~ ., data = train_data[train, ], nvmax = 30)

val.errors = rep(NA, 30)
for (i in 1:30) {
  pred = predict.regsubsets(regfit.fwd2, train_data[-train, ], id = i)
  val.errors[i] = mean((train_data$price[-train] - pred) ^ 2)
}

plot(sqrt(val.errors), ylab = "Root MSE", pch = 20, type = "b")
points(sqrt(regfit.fwd2$rss / length(train)), col = "blue", pch = 20, type = "b")
legend("topright", legend = c("Training", "Validation"), col = c("blue", "black"), pch = 20)

which.min(sqrt(val.errors))

# Cross-validation setup
set.seed(11)
folds = sample(rep(1:10, length = nrow(train_data)))

nvmax_value <- min(30, ncol(train_data) - 1)
cv.errors = matrix(NA, 10, nvmax_value)
train.errors = matrix(NA, 10, nvmax_value)

for (k in 1:10) {
  best.fit = regsubsets(price ~ ., data = train_data[folds != k, ], nvmax = nvmax_value)
  
  # Get the actual number of models fitted
  available_models <- length(best.fit$rss)
  
  # Ensure `available_models` does not exceed `nvmax_value`
  available_models <- min(available_models, nvmax_value)
  
  # Assign RSS values only for available models
  train.errors[k, 1:available_models] <- best.fit$rss[1:available_models] / dim(train_data[folds != k, ])[1]
  
  # Ensure prediction only for available models
  for (i in 1:available_models) {  
    if (length(coef(best.fit, i)) > 0) {  
      pred = predict.regsubsets(best.fit, train_data[folds == k, ], id = i)
      cv.errors[k, i] = mean((train_data$price[folds == k] - pred) ^ 2)
    }
  }
}

rmse.cv = sqrt(apply(cv.errors, 2, mean))
rmse.train = sqrt(apply(train.errors, 2, mean))

#any(is.na(rmse.train))  # Should return FALSE
#print(rmse.train)       # Check the values

plot(rmse.cv, pch = 20, type = "b", col = "black", 
     ylim = range(c(rmse.cv, rmse.train)), ylab = "RMSE", xlab = "Number of Variables")
points(rmse.train, col = "blue", pch = 4, type = "b", lty = 2, lwd = 2)
legend("topright", legend = c("Training", "10-Fold CV"), col = c("blue", "black"), pch = c(4, 20), lty = c(2, 1))


# Final model selection
regfit.final = regsubsets(price ~ ., data = train_data, nvmax = 30)
max_model_size <- length(regfull.summary$cp)
max_model_size

if (30 <= max_model_size) {
  coef(regfit.final, 30)
} else {
  print("Model size 30 is not available. Try using a different value.")
}

# Get the number of variables in the final model (30 in your case)
final_model_size <- 30

# Training Predictions
train_pred <- predict.regsubsets(regfit.final, train_data, id = final_model_size)

# Training RMSE
train_rmse <- sqrt(mean((train_data$price - train_pred)^2))
print(paste("Training RMSE:", train_rmse))

# Training R-squared
ss_total_train <- sum((train_data$price - mean(train_data$price))^2)
ss_residual_train <- sum((train_data$price - train_pred)^2)
train_r2 <- 1 - (ss_residual_train / ss_total_train)
print(paste("Training R²:", train_r2))

# Test Predictions
test_pred <- predict.regsubsets(regfit.final, test_data, id = final_model_size)

# Test MSE
test_mse <- mean((test_data$price - test_pred)^2)
print(paste("Test MSE:", test_mse))

# Test RMSE
test_rmse <- sqrt(test_mse)
print(paste("Test RMSE:", test_rmse))

# Test R-squared
ss_total_test <- sum((test_data$price - mean(test_data$price))^2)
ss_residual_test <- sum((test_data$price - test_pred)^2)
test_r2 <- 1 - (ss_residual_test / ss_total_test)
print(paste("Test R²:", test_r2))

plot(train_pred, train_data$price - train_pred, 
     xlab = "Fitted Values", ylab = "Residuals", 
     main = "Residuals vs. Fitted Values", col="blue")
abline(h = 0, col = "red")

par(mfrow = c(1, 2))
hist(train_data$price - train_pred, 
     breaks = 30, 
     main = "Histogram of Residuals", 
     col = "lightblue", 
     xlab = "Residuals", 
     ylab = "Frequency")

qqnorm(train_data$price - train_pred)
qqline(train_data$price - train_pred, col = "red")
par(mfrow = c(1, 1))

library(car)
vif_values <- vif(lm(price ~ ., data = train_data))
print(vif_values)

library(lmtest)
model_lm <- lm(price ~ ., data = train_data)
dw_test <- dwtest(model_lm)
print(dw_test)



