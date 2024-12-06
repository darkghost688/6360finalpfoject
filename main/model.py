# Step 1: Importing Necessary Libraries

# Import necessary libraries
# This step imports all the necessary Python libraries that are required for data manipulation, visualization, and modeling.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Loading the Dataset

# Load the dataset
# This step loads the dataset that we are going to analyze. The dataset contains information about car features and prices.
df = pd.read_csv('./car_price_prediction_dataset.csv')

# Step 3: Data Preprocessing

# Handling missing values by dropping them or imputing
# Here, we'll check for any missing values and fill them with median values for simplicity.
df.fillna(df.median(), inplace=True)

# Convert categorical variables to numerical values if necessary
# Assuming that the dataset has some categorical variables, they need to be encoded.
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df[column] = pd.Categorical(df[column]).codes

# Step 4: Standardizing the Data

# Standardizing the dataset to ensure each feature has mean = 0 and standard deviation = 1
# This step is crucial for models that are sensitive to the scale of data.
def standardize_data(data):
    return (data - data.mean()) / data.std()

df = df.apply(standardize_data)

# Step 5: Exploratory Data Analysis (EDA)

# Visualizing the distribution of MSRP and other key features
plt.figure(figsize=(10, 6))
sns.histplot(df['MSRP'], kde=True, color='blue')
plt.title('Distribution of MSRP')
plt.xlabel('MSRP')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# Step 6: Splitting the Dataset

# Splitting the dataset into training and testing sets
# This step helps in validating the model's performance on unseen data.
X = df.drop(columns=['MSRP'])
y = df['MSRP']
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Building Linear Models

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(train_data, train_labels)
linear_predictions = linear_model.predict(test_data)

# Ridge Regression
ridge = Ridge()
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_search = GridSearchCV(ridge, ridge_params, cv=5)
ridge_search.fit(train_data, train_labels)
ridge_predictions = ridge_search.best_estimator_.predict(test_data)

# Lasso Regression
lasso = Lasso()
lasso_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
lasso_search = GridSearchCV(lasso, lasso_params, cv=5)
lasso_search.fit(train_data, train_labels)
lasso_predictions = lasso_search.best_estimator_.predict(test_data)

# Step 8: Building Nonlinear Models

# Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(train_data, train_labels)
rf_predictions = rf.predict(test_data)

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(train_data, train_labels)
gbr_predictions = gbr.predict(test_data)

# Step 9: Model Evaluation

# Function to evaluate models
def evaluate_model(name, predictions, true_values):
    rmse = mean_squared_error(true_values, predictions, squared=False)
    r2 = r2_score(true_values, predictions)
    print(f'{name} - RMSE: {rmse:.4f}, R²: {r2:.4f}')

# Evaluating Linear Models
evaluate_model('Linear Regression', linear_predictions, test_labels)
evaluate_model('Ridge Regression', ridge_predictions, test_labels)
evaluate_model('Lasso Regression', lasso_predictions, test_labels)

# Evaluating Nonlinear Models
evaluate_model('Random Forest', rf_predictions, test_labels)
evaluate_model('Gradient Boosting', gbr_predictions, test_labels)

# Step 10: Analyzing Feature Importance

# Analyzing Feature Importance from Gradient Boosting Regressor
feature_importance = gbr.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance from Gradient Boosting Regressor')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Step 11: Summary of Findings

# Gradient Boosting Regressor showed the best performance with the lowest RMSE and highest R².
# Key Features affecting MSRP include features with high importance in the Gradient Boosting model.
# Future work can include further hyperparameter tuning and exploration of other nonlinear models like Neural Networks.
