#s16383

library(tidyverse)   # For data manipulation and visualization
library(caret)  # For machine learning workflows
library(xgboost) 
library(rpart) 
library(scales)
library(glmnet)

flight_data <- read.csv("Clean_Dataset.csv")

flight_data$airline <- as.factor(flight_data$airline)
flight_data$source_city <- as.factor(flight_data$source_city)
flight_data$stops <- as.factor(flight_data$stops)
flight_data$destination_city <- as.factor(flight_data$destination_city)
flight_data$class <- as.factor(flight_data$class)
flight_data$departure_time <- as.factor(flight_data$departure_time)
flight_data$arrival_time <- as.factor(flight_data$arrival_time)

flight_data_clean=flight_data
dummy_vars <- dummyVars(" ~ airline + source_city + stops + destination_city + 
                         class + departure_time + arrival_time", 
                        data = flight_data_clean, 
                        fullRank = TRUE)
dummy_data <- predict(dummy_vars, newdata = flight_data_clean)
flight_data_encoded <- cbind(flight_data_clean[, c("duration", "days_left", "price")], 
                             as.data.frame(dummy_data))

correlations <- cor(flight_data_encoded$price, flight_data_encoded[, -which(names(flight_data_encoded) == "price")])
correlation_df <- data.frame(
  Feature = colnames(flight_data_encoded)[2:ncol(flight_data_encoded)],
  Correlation = as.numeric(correlations)
)
correlation_df <- correlation_df[order(abs(correlation_df$Correlation), decreasing = TRUE), ]
head(correlation_df, 10)  # Top 10 correlations

set.seed(123)  # For reproducibility

train_indices <- createDataPartition(flight_data_encoded$price, p = 0.7, list = FALSE)
train_data <- flight_data_encoded[train_indices, ]

test_data <- flight_data_encoded[-train_indices, ]
X_train <- train_data[, names(train_data) != "price"]
y_train <- train_data$price
X_test <- test_data[, names(test_data) != "price"]
y_test <- test_data$price

preprocess_params <- preProcess(X_train[, c("duration", "days_left")], method = c("range"))
X_train_scaled <- predict(preprocess_params, X_train)
X_test_scaled <- predict(preprocess_params, X_test)

# Function to calculate evaluation metrics
calculate_metrics <- function(actual, predicted) {
  mse <- mean((predicted - actual)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(predicted - actual))
  
  # Calculate R-squared
  ss_total <- sum((actual - mean(actual))^2)
  ss_residual <- sum((actual - predicted)^2)
  r_squared <- 1 - (ss_residual / ss_total)
  
  # Calculate Adjusted R-squared
  n <- length(actual)
  p <- ncol(X_train)  # Number of predictors
  adj_r_squared <- 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
  
  return(list(
    MSE = mse,
    RMSE = rmse,
    MAE = mae,
    R_squared = r_squared,
    Adjusted_R_squared = adj_r_squared
  ))
}
# Initialize results dataframe
model_results <- data.frame(
  Model = character(),
  MSE = numeric(),
  RMSE = numeric(), 
  MAE = numeric(),
  Adjusted_R_squared = numeric(),
  stringsAsFactors = FALSE
)
# 6. XGBoost Regressor
set.seed(123)
xgb_model <- train(
  x = as.matrix(X_train_scaled),
  y = y_train,
  method = "xgbTree",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = expand.grid(
    nrounds = 100,
    max_depth = 6,
    eta = 0.3,
    gamma = 0,
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = 1
  )
)
xgb_predictions <- predict(xgb_model, as.matrix(X_test_scaled))
xgb_metrics <- calculate_metrics(y_test, xgb_predictions)
model_results <- rbind(model_results, data.frame(
  Model = "XGBoost Regressor",
  MSE = xgb_metrics$MSE,
  RMSE = xgb_metrics$RMSE,
  MAE = xgb_metrics$MAE,
  Adjusted_R_squared = xgb_metrics$Adjusted_R_squared
))

# Sort results by Adjusted R-squared (descending)
model_results <- model_results %>%
  arrange(desc(Adjusted_R_squared))

# Format metrics for display
model_results$MSE <- format(model_results$MSE, scientific = TRUE, digits = 3)
model_results$RMSE <- round(model_results$RMSE, 2)
model_results$MAE <- round(model_results$MAE, 2)
model_results$Adjusted_R_squared <- round(model_results$Adjusted_R_squared, 4)

# Display the results table
print(model_results)

best_model <- xgb_model  
best_predictions <- predict(best_model, as.matrix(X_test_scaled))
prediction_df <- data.frame(
  Actual = y_test,
  Predicted = best_predictions
)
ggplot(prediction_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Actual vs Predicted Flight Prices",
    x = "Actual Price",
    y = "Predicted Price"
  ) +
  theme_minimal() +
  annotate("text", x = max(y_test) * 0.7, y = max(best_predictions) * 0.3,
           label = paste("Adjusted R²:", round(xgb_metrics$Adjusted_R_squared, 4)),
           size = 4)

# Plot price vs days left for departure
days_data <- data.frame(
  days_left = test_data$days_left,
  actual_price = y_test,
  predicted_price = best_predictions
)

# Calculate average prices by days left
days_summary <- days_data %>%
  group_by(days_left) %>%
  summarize(
    avg_actual = mean(actual_price),
    avg_predicted = mean(predicted_price)
  ) %>%
  arrange(days_left)

# Create the days vs price plot
ggplot(days_summary, aes(x = days_left)) +
  geom_line(aes(y = avg_actual, color = "Actual"), linewidth = 1) +
  geom_line(aes(y = avg_predicted, color = "Predicted"), linewidth = 1) +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  labs(
    title = "Days Left vs Flight Price",
    x = "Days Left Before Departure",
    y = "Average Price",
    color = "Price Type"
  ) +
  theme_minimal()
par(mfrow = c(1, 1))

# Step 5: Feature Importance (using XGBoost)
# Extract feature importance
importance_matrix <- xgb.importance(
  feature_names = colnames(X_train_scaled),
  model = xgb_model$finalModel
)

# Plot feature importance
xgb.plot.importance(importance_matrix[1:15,], top_n = 10)

2# Summary and Conclusion
cat("\nSummary of Flight Price Prediction Analysis:\n")
cat("Best performing model: ", model_results$Model[1], "\n")
cat("With Adjusted R-squared of: ", model_results$Adjusted_R_squared[1], "\n")
cat("RMSE: ", model_results$RMSE[1], "\n")
cat("MAE: ", model_results$MAE[1], "\n\n")
cat("This matches the findings from the paper, which identified XGBoost as the best performer.\n")

