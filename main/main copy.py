{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Step 1: Importing Necessary Libraries\n",
    "\n",
    "# Import necessary libraries\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import networkx as nx\n",
    "from pgmpy.estimators import HillClimbSearch, BicScore\n",
    "from pgmpy.models import BayesianModel\n",
    "from pgmpy.estimators import BayesianEstimator\n",
    "from pgmpy.inference import VariableElimination"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 2: Loading the Dataset\n",
    "\n",
    "# Load the dataset\n",
    "# Dataset link: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset\n",
    "df = pd.read_csv('./diabetes_prediction_dataset.csv')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 3: Data Preprocessing\n",
    "\n",
    "# Handling missing values by dropping them or imputing\n",
    "# Here, we'll check for any missing values and fill them with median values for simplicity.\n",
    "df.fillna(df.median(), inplace=True)\n",
    "\n",
    "# Convert categorical variables to numerical values if necessary\n",
    "# Assuming that the dataset has some categorical variables, they need to be encoded.\n",
    "categorical_columns = df.select_dtypes(include=['object']).columns\n",
    "for column in categorical_columns:\n",
    "    df[column] = pd.Categorical(df[column]).codes\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 4: Standardizing the Data\n",
    "\n",
    "# Standardizing the dataset to ensure each feature has mean = 0 and standard deviation = 1\n",
    "def standardize_data(data):\n",
    "    return (data - data.mean()) / data.std()\n",
    "\n",
    "df = df.apply(standardize_data)\n",
    "\n",
    "# Step 5: Exploratory Data Analysis (EDA)\n",
    "\n",
    "# Visualizing the distribution of key features\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.histplot(df['BMI'], kde=True, color='blue')\n",
    "plt.title('Distribution of BMI')\n",
    "plt.xlabel('BMI')\n",
    "plt.ylabel('Frequency')\n",
    "plt.show()\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.countplot(x='Outcome', data=df, palette='viridis')\n",
    "plt.title('Distribution of Diabetes Outcome')\n",
    "plt.xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')\n",
    "plt.ylabel('Count')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 6: Building the Bayesian Network\n",
    "\n",
    "# We'll use the Hill Climb Search Algorithm to find the optimal structure for the Bayesian Network\n",
    "# Score-based method (BIC) to build the graph structure\n",
    "hc = HillClimbSearch(df, scoring_method=BicScore(df))\n",
    "model = hc.estimate()\n",
    "\n",
    "# Visualize the structure of the learned Bayesian Network\n",
    "plt.figure(figsize=(12, 10))\n",
    "G = nx.DiGraph(model.edges())\n",
    "nx.draw_networkx(G, with_labels=True, node_size=2000, node_color='skyblue', font_size=15, font_weight='bold')\n",
    "plt.title('Bayesian Network Structure')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 7: Creating the Bayesian Model\n",
    "\n",
    "# Create the Bayesian Model using pgmpy\n",
    "bayesian_model = BayesianModel(model.edges())\n",
    "\n",
    "# Fit the model with data using Bayesian Estimator\n",
    "bayesian_model.fit(df, estimator=BayesianEstimator, prior_type='BDeu')\n",
    "\n",
    "# Step 8: Inference using Bayesian Network\n",
    "\n",
    "# We can now perform inference using the Bayesian Network to understand causal relationships\n",
    "inference = VariableElimination(bayesian_model)\n",
    "\n",
    "# Example Query\n",
    "# What is the probability of having diabetes given that BMI is above average?\n",
    "bmi_above_average = df['BMI'].mean() + 1  # Taking one standard deviation above the mean as the threshold\n",
    "query_result = inference.query(variables=['Outcome'], evidence={'BMI': bmi_above_average})\n",
    "\n",
    "print('Probability of having diabetes given BMI is above average:', query_result)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 9: Analyzing the Results\n",
    "\n",
    "# Discussion on the causal relationships identified by the Bayesian Network\n",
    "# The learned Bayesian Network should give us insight into which variables influence diabetes the most.\n",
    "# Based on the structure of the network, we can look for nodes that have a direct link to the 'Outcome' variable.\n",
    "# This helps in understanding which variables have a strong direct causal influence.\n",
    "\n",
    "# Key Insights:\n",
    "# - Use the Bayesian Network structure to see which features are directly linked to the outcome (diabetes)\n",
    "# - The network edges indicate causal relationships between variables, helping to determine which features have a potential cause-and-effect relationship with diabetes.\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
