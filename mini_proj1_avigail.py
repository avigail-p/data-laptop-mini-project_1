#Mini project 1= Avigail Cohen 214122871


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Task 1-Plot the price of all the laptops

sns.set_theme(style="whitegrid") # Set a clean and readable visual style for all plots

df = pd.read_csv("laptop_price - dataset.csv") # Load the laptop price dataset into a pandas DataFrame
"""
plt.figure(figsize=(12,5)) # Create a new figure with a defined size
# Plot the price of each laptop as a single point
# x-axis: laptop index (order in the dataset)
# y-axis: laptop price in Euro
sns.scatterplot(
    x=df.index,
    y=df["Price (Euro)"],
    alpha=0.6,   # Set transparency to reduce overlap between points
    s=40         # Set the size of each point
)
plt.title("Prices of All Laptops")
plt.xlabel("Laptop Index")
plt.ylabel("Price (Euro)")
plt.show()

#Task 2

avg_price_by_company = df.groupby("Company")["Price (Euro)"].mean()# Group the data by company and calculate the mean price
avg_price_by_company = avg_price_by_company.sort_values(ascending=False) # Sort the companies by average price in descending order
# Identify the company with the highest average price
most_expensive_company = avg_price_by_company.index[0]
highest_avg_price = avg_price_by_company.iloc[0]
print(f"The company with the highest average laptop price is {most_expensive_company}, with an average price of {highest_avg_price:.2f} Euro.")

plt.figure(figsize=(10,6))

sns.barplot(
    x=avg_price_by_company.values,
    y=avg_price_by_company.index
)

plt.title("Average Laptop Price by Company")
plt.xlabel("Average Price (Euro)")
plt.ylabel("Company")
plt.show() #show the plot

#task 3- Find the different types of Operating systems present in the data (OpSys), and make them uniform.

df["OpSys_clean"] = df["OpSys"]# Create a cleaned operating system column
# Unify Mac operating system names
df["OpSys_clean"] = df["OpSys_clean"].replace(
    ["macOS", "Mac OS X"],
    "macOS"
)
# Unify Windows 10 variants

df["OpSys_clean"] = df["OpSys_clean"].replace(
    ["Windows 10", "Windows 10 S"],
    "Windows 10"
)
print(df["OpSys_clean"].unique()) # Check the cleaned unique values

#Task 4-Plot price distribution for each operating system

# Get unique operating systems
operating_systems = df["OpSys_clean"].unique()

# Loop through each operating system
for os_name in operating_systems:
    
    # Create a new figure for each OS
    plt.figure(figsize=(8,5))
    
    # Plot the price distribution
    sns.histplot(
        df[df["OpSys_clean"] == os_name]["Price (Euro)"],
        bins=25,
    )
    
    # Add titles and labels
    plt.title(f"Price Distribution for {os_name}")
    plt.xlabel("Price (Euro)")
    plt.ylabel("Count")
    
    # Show the plot
    plt.show()
"""
# Task 5: Relationship between RAM and price

# Task 5: Relationship between RAM and laptop price
# This task examines the relationship between RAM size and laptop price
# and identifies outliers based on deviation from the regression line.

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Relationship between RAM and price
# -------------------------------

# Define variables
x = df["RAM (GB)"]
y = df["Price (Euro)"]

# Fit a linear regression model
slope, intercept = np.polyfit(x, y, 1)

# Predicted price based on RAM
df["predicted_price"] = intercept + slope * x

# -------------------------------
# Step 2: Outlier detection using residuals
# -------------------------------

# Residual = actual price - predicted price
df["residual"] = y - df["predicted_price"]

# Standard deviation of residuals
residual_std = df["residual"].std()

# Define outliers as points more than 2 SD from the regression line
df["outlier"] = abs(df["residual"]) > 2 * residual_std

# Print detected outliers
print("Number of outliers:", df["outlier"].sum())
print(df[df["outlier"]][["RAM (GB)", "Price (Euro)", "residual"]])

# -------------------------------
# Step 3: Visualization
# -------------------------------

plt.figure(figsize=(8, 6))

# Regular observations
plt.scatter(
    df.loc[~df["outlier"], "RAM (GB)"],
    df.loc[~df["outlier"], "Price (Euro)"],
    alpha=0.6,
    label="Within 2 SD from regression"
)

# Outliers
plt.scatter(
    df.loc[df["outlier"], "RAM (GB)"],
    df.loc[df["outlier"], "Price (Euro)"],
    color="orange",
    s=50,
    label="Outliers"
)

# Regression line
x_sorted = np.sort(x)
plt.plot(
    x_sorted,
    intercept + slope * x_sorted,
    color="red",
    label="Regression line"
)

plt.xlabel("RAM (GB)")
plt.ylabel("Price (Euro)")
plt.title("Relationship Between RAM and Laptop Price")
plt.legend()
plt.show()
"""
# Task 6: Extract storage type from the Memory column

#A new column named "Storage type" was created by extracting the storage type information from the "Memory" column. Using a regular expression, the storage medium (SSD or HDD) was identified for each laptop, allowing further analysis based on storage technology.

# Extract the storage type (SSD or HDD) from the Memory column
# The regular expression searches for the keywords 'SSD' or 'HDD'
df["Storage type"] = df["Memory"].str.extract(r'(SSD|HDD|Flash Storage|Hybrid)')
# Uncomment the line below to verify the extraction result
# print(df[["Memory", "Storage type"]].head())



"""
