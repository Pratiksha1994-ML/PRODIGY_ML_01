# Prodigy InfoTech Internship Task 01
# House Price Prediction using Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset from folder
data = pd.read_csv("house-prices-advanced-regression-techniques/train.csv")

# Display first 5 rows
print("Dataset Preview:")
print(data.head())

# Select features
# GrLivArea = Square Footage
# BedroomAbvGr = Bedrooms
# FullBath = Bathrooms
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# Target variable
y = data['SalePrice']

# Check for missing values
print("\nMissing Values:")
print(X.isnull().sum())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Example prediction
# Example house:
# 2000 sq ft, 3 bedrooms, 2 bathrooms
sample_house = [[2000, 3, 2]]

predicted_price = model.predict(sample_house)

print("\nSample House Prediction:")
print("Square Footage: 2000")
print("Bedrooms: 3")
print("Bathrooms: 2")
print("Predicted House Price: $", round(predicted_price[0], 2))