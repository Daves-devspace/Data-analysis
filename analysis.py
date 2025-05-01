#1: Load and Explore the Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
except Exception as e:
    print("Error loading dataset:", e)

# Show first few rows
print(df.head())

# Check info
print(df.info())

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# No cleaning needed for Iris dataset, but typically:
# df = df.dropna() or df.fillna(method='ffill')

#2: Basic Data Analysis

# Basic stats
print(df.describe())

# Group by species and calculate mean of numerical values
grouped = df.groupby('target').mean()
print(grouped)

# Add species names for clarity
df['species'] = df['target'].apply(lambda x: iris.target_names[x])
grouped_by_species = df.groupby('species').mean()
print(grouped_by_species)

#3:Data Visualizations
# Line chart: Mean petal length over index (not time-series, for example only)
plt.plot(df['petal length (cm)'][:30])
plt.title("Petal Length Over Index (Sample)")
plt.xlabel("Sample Index")
plt.ylabel("Petal Length (cm)")
plt.show()

# Bar chart: Average petal length per species
grouped_by_species['petal length (cm)'].plot(kind='bar', color='skyblue')
plt.title("Average Petal Length by Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram: Sepal width
plt.hist(df['sepal width (cm)'], bins=10, color='orange', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot: Sepal Length vs Petal Length
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Sepal vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()
