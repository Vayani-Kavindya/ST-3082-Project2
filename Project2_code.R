library(tidyverse)
library(caret)
library(dplyr)

setwd("D:/university/Level 3/2nd Sem/ST 3082/Project2")

flight_data <- read.csv("Clean_Dataset.csv")
flight_data <- flight_data %>% select(-c(X, flight))

# Split the data into training and testing sets 
set.seed(123)
train_indices <- createDataPartition(flight_data$price, p = 0.8, list = FALSE)
train_data <- flight_data[train_indices, ]
test_data <- flight_data[-train_indices, ]

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



######################################### Findings of Descriptive Analysis #########################################################

library(tidyverse)
library(caret)
library(dplyr)
library(e1071) # For skewness

# Basic Descriptive Statistics (Post-Preprocessing)
summary(train_data_processed)

# Distribution of the price
hist(train_data$price,prob = TRUE,xlab = 'Price',main = 'Distribution of the Price',col = 'lightblue')
boxplot(train_data$price,col = 'lightblue',main = "Boxplot of Price", ylab = "Price")
grid()

# Skewness of Price
skewness_price <- skewness(train_data_processed$price)
print(paste("Skewness of Price:", skewness_price))

# Identifying Multicollinearity
predictor_correlation_matrix <- cor(train_data_processed[, -which(names(train_data_processed) %in% c("price"))])
corr_df <- as.data.frame(as.table(predictor_correlation_matrix))
colnames(corr_df) <- c("Variable1", "Variable2", "Correlation")

high_corr_df <- corr_df %>%
  filter(abs(Correlation) > 0.3 & Variable1 != Variable2) %>%
  arrange(desc(abs(Correlation)))

print(high_corr_df)

# Identify highly correlated pairs
high_correlations <- which(abs(predictor_correlation_matrix) > 0.5 & abs(predictor_correlation_matrix) < 1, arr.ind = TRUE) # Adjust threshold as needed
if (nrow(high_correlations) > 0) {
  for (i in 1:nrow(high_correlations)) {
    row_index <- high_correlations[i, 1]
    col_index <- high_correlations[i, 2]
    var1 <- colnames(predictor_correlation_matrix)[row_index]
    var2 <- colnames(predictor_correlation_matrix)[col_index]
    correlation_value <- predictor_correlation_matrix[row_index, col_index]
    print(paste("Correlation:", var1, "and", var2, "Correlation:", correlation_value))
  }
} else {
  print("No highly correlated predictor variables found (correlation > 0.5).")
}



#### FAMD test

data_indices <- createDataPartition(train_data$price, p = 1, list = FALSE)
data <- flight_data[data_indices, ]

#yc is the output variable of training data
#xc is the predictor variables in training data

Xc=data[,c(-10)]
Yc=data[10]

head(Xc)  # View first few rows
head(Yc)

library(FactoMineR)
library(factoextra)
library(dplyr)

# Ensure categorical variables are factors
categorical_vars <- names(Xc)[sapply(Xc, is.character)]
Xc[categorical_vars] <- lapply(Xc[categorical_vars], as.factor)


Xc <- Xc[!duplicated(Xc), ]
famd_data <- Xc %>%
  select(starts_with("log_"), where(is.factor))

# Run FAMD
famd_result <- FAMD(famd_data, ncp = 5, graph = FALSE)
fviz_screeplot(famd_result, addlabels = TRUE)

anyDuplicated(rownames(famd_result$ind$coord))
rownames(famd_result$ind$coord) <- make.names(rownames(famd_result$ind$coord), unique = TRUE)


# Extract FAMD coordinates
scores <- as.data.frame(famd_result$ind$coord)
scores$ID <- rownames(scores)  
library(ggplot2)

scores <- as.data.frame(famd_result$ind$coord)
scores$ID <- rownames(scores)

# Plot the first two dimensions
ggplot(scores, aes(x = Dim.1, y = Dim.2)) +
  geom_point(color = "blue", alpha = 0.7) +  # Scatter plot of points
  labs(title = "FAMD Score Plot",
       x = "Dim1 (6.2%)",
       y = "Dime2 (5.2%)") +
  theme_minimal()




################################## Model fitting #########################################


######################## Lasso Model Fitting ########################

library(glmnet)

# Fitting Lasso Regression
fit.lasso <- glmnet(as.matrix(X_train), y_train, alpha = 1) # alpha = 1 for Lasso
plot(fit.lasso, xvar = "lambda", label = TRUE, lw = 2)

# Cross-validation for best lambda
cv.lasso <- cv.glmnet(as.matrix(X_train), y_train, alpha = 1) # alpha = 1 for Lasso
plot(cv.lasso)
bestlam <- cv.lasso$lambda.min
bestlam

# Coefficients with best lambda
coef(fit.lasso, s = bestlam)

y_pred_train <- predict(fit.lasso, s = bestlam, newx = as.matrix(X_train))

# Training MSE
train_Rmse <- sqrt(mean((y_train - y_pred_train)^2))
print(paste("Training RMSE:", train_Rmse))

# Training R^2
ss_res <- sum((y_train - y_pred_train)^2)
ss_tot <- sum((y_train - mean(y_train))^2)
train_r2 <- 1 - (ss_res / ss_tot)
print(paste("Training R^2:", train_r2))

# Test set predictions
y_pred_test <- predict(fit.lasso, s = bestlam, newx = as.matrix(X_test))

# Test MSE
test_rmse <- sqrt(mean((y_test - y_pred_test)^2))
print(paste("Test RMSE:", test_rmse))

# Test R^2
ss_res <- sum((y_test - y_pred_test)^2)
ss_tot <- sum((y_test - mean(y_test))^2)
test_r2 <- 1 - (ss_res / ss_tot)
print(paste("Test R^2:", test_r2))






######################## Elastic Net Model Fitting ########################

# Cross-validation for best lambda and alpha
cv.elastic <- cv.glmnet(as.matrix(X_train), y_train, alpha = 0.5) # alpha = 0.5 for Elastic Net (you can tune this)
plot(cv.elastic)
bestlam_elastic <- cv.elastic$lambda.min
bestlam_elastic

# Fitting Elastic Net Regression
fit.elastic <- glmnet(as.matrix(X_train), y_train, alpha = 0.5, lambda = bestlam_elastic)

# Coefficients with best lambda
coef(fit.elastic, s = bestlam_elastic)

y_pred_train_elastic <- predict(fit.elastic, s = bestlam_elastic, newx = as.matrix(X_train))

# Training MSE
train_rmse_elastic <- sqrt(mean((y_train - y_pred_train_elastic)^2))
print(paste("Elastic Net Training RMSE:", train_rmse_elastic))

# Training R^2
ss_res_elastic <- sum((y_train - y_pred_train_elastic)^2)
ss_tot_elastic <- sum((y_train - mean(y_train))^2)
train_r2_elastic <- 1 - (ss_res_elastic / ss_tot_elastic)
print(paste("Elastic Net Training R^2:", train_r2_elastic))

# Test set predictions
y_pred_test_elastic <- predict(fit.elastic, s = bestlam_elastic, newx = as.matrix(X_test))

# Test MSE
test_rmse_elastic <- sqrt(mean((y_test - y_pred_test_elastic)^2))
print(paste("Elastic Net Test MSE:", test_rmse_elastic))

# Test R^2
ss_res_test_elastic <- sum((y_test - y_pred_test_elastic)^2)
ss_tot_test_elastic <- sum((y_test - mean(y_test))^2)
test_r2_elastic <- 1 - (ss_res_test_elastic / ss_tot_test_elastic)
print(paste("Elastic Net Test R^2:", test_r2_elastic))

























