from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)


# Create boxplot figure
df.boxplot(figsize=(12, 6))

plt.title("California Housing Dataset Boxplot")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()