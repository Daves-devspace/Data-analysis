# Python Data Analysis Assignment
# ========================
# Objectives:
# - Load and analyze a dataset using pandas.
# - Create simple plots with matplotlib for visualization.

# Task 1: Load and Explore the Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Use seaborn's 'flights' dataset (passengers over time)
try:
    df = sns.load_dataset('flights')
    print("Dataset loaded successfully.\n")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Display first few rows
print("First five rows:")
print(df.head(), "\n")

# Check data types and missing values
print("Data types ..:")
print(df.dtypes, "\n")

print("Missing values per column:")
print(df.isnull().sum(), "\n")

# Clean dataset (drop any missing values, if present)
df_clean = df.dropna()
print(f"Shape after dropping missing values: {df_clean.shape}\n")

# Task 2: Basic Data Analysis

# 1. Summary statistics for numerical columns
print("Summary statistics:")
print(df_clean.describe(), "\n")

# 2. Group by year and compute mean passengers
yearly_mean = df_clean.groupby('year')['passengers'].mean()
print("Mean passengers per year:")
print(yearly_mean, "\n")

# Task 3: Data Visualization

# 1. Line chart: passengers over time
# Convert 'year' and 'month' into a datetime for plotting
df_clean['date'] = pd.to_datetime(
    df_clean['year'].astype(str) + '-' + df_clean['month'],
    format='%Y-%b'
)

plt.figure()
plt.plot(df_clean['date'], df_clean['passengers'], linewidth=2)
plt.title('Monthly Air Passengers Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.tight_layout()
plt.grid(True)
plt.show()

# 2. Bar chart: average passengers by month
monthly_mean = df_clean.groupby('month')['passengers'].mean()
plt.figure()
monthly_mean.reindex([
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]).plot(kind='bar')
plt.title('Average Passengers per Month (1949-1960)')
plt.xlabel('Month')
plt.ylabel('Average Number of Passengers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Histogram: distribution of passenger counts
plt.figure()
plt.hist(df_clean['passengers'], bins=12)
plt.title('Distribution of Monthly Passenger Counts')
plt.xlabel('Number of Passengers')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Scatter plot: year vs passengers
plt.figure()
plt.scatter(df_clean['year'], df_clean['passengers'], alpha=0.7)
plt.title('Year vs. Passenger Count')
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.tight_layout()
plt.show()

# Observations:
# - The line chart shows a clear upward trend in air passengers from 1949 to 1960.
# - Seasonal patterns are visible: peaks in mid-year months (June-August).
# - The bar chart confirms higher average passenger counts in summer months.
# - The histogram reveals a right-skewed distribution.
# - The scatter plot highlights year-over-year growth with some variability in counts.

