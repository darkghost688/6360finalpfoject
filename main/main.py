# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

# Load the dataset
# Dataset link: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
df = pd.read_csv('./diabetes_prediction_dataset.csv')

# Step 1: Data Preprocessing
# Handling missing values by dropping them or imputing
# Here, we'll check for any missing values and fill them with median values for simplicity.
df.fillna(df.median(), inplace=True)

# Convert categorical variables to numerical values if necessary
# Assuming that the dataset has some categorical variables, they need to be encoded.
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df[column] = pd.Categorical(df[column]).codes

# Step 2: Standardize the data
# Standardizing the dataset to ensure each feature has mean = 0 and standard deviation = 1
def standardize_data(data):
    return (data - data.mean()) / data.std()

df = df.apply(standardize_data)

# Step 3: Building the Bayesian Network
# We'll use the Hill Climb Search Algorithm to find the optimal structure for the Bayesian Network
# Score-based method (BIC) to build the graph structure
hc = HillClimbSearch(df, scoring_method=BicScore(df))
model = hc.estimate()

# Visualize the structure of the learned Bayesian Network
plt.figure(figsize=(10, 8))
G = nx.DiGraph(model.edges())
nx.draw_networkx(G, with_labels=True, node_size=2000, node_color='skyblue', font_size=15, font_weight='bold')
plt.title('Bayesian Network Structure')
plt.show()

# Step 4: Create the Bayesian Model using pgmpy
bayesian_model = BayesianModel(model.edges())

# Fit the model with data using Bayesian Estimator
bayesian_model.fit(df, estimator=BayesianEstimator, prior_type='BDeu')

# Step 5: Inference using Bayesian Network
# We can now perform inference using the Bayesian Network to understand causal relationships
inference = VariableElimination(bayesian_model)

# Example Query
# What is the probability of having diabetes given that BMI is above average?
bmi_above_average = df['BMI'].mean() + 1  # Taking one standard deviation above the mean as the threshold
query_result = inference.query(variables=['Outcome'], evidence={'BMI': bmi_above_average})

print('Probability of having diabetes given BMI is above average:', query_result)

# Step 6: Analyzing the Results
# Discussion on the causal relationships identified by the Bayesian Network
# The learned Bayesian Network should give us insight into which variables influence diabetes the most.
# Based on the structure of the network, we can look for nodes that have a direct link to the 'Outcome' variable.
# This helps in understanding which variables have a strong direct causal influence.

# Key Insights:
# - Use the Bayesian Network structure to see which features are directly linked to the outcome (diabetes)
# - The network edges indicate causal relationships between variables, helping to determine which features have a potential cause-and-effect relationship with diabetes.
