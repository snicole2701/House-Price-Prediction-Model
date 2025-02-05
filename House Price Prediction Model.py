import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv('C:/Users/USER-PC/Desktop/Code Folder/House Price Prediction Model/House Prices Dataset.csv')

# View the data
dataset

# Identify if we have any null values
dataset.info()

# Very few NA values so we will drop them from the dataset
dataset.dropna(inplace=True)

# Split the data into train and test sets
from sklearn.model_selection import train_test_split

X = dataset.drop(['median_house_value'], axis=1)
y = dataset['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
train_data = X_train.join(y_train)

# Select only numeric columns
numeric_cols = train_data.select_dtypes(include=[float, int])
corr_matrix = numeric_cols.corr()

# Plot heatmap of the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
plt.show()

# Apply logarithmic transformation to reduce skewness
train_data['total_rooms'] = np.log(train_data['total_rooms']) + 1
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms']) + 1
train_data['population'] = np.log(train_data['population']) + 1
train_data['households'] = np.log(train_data['households']) + 1

# Plot histograms of transformed features
train_data.hist()
plt.show()

# We want to include the column that currently has string values in the model and therefore need to change the string to a type that can be used
train_data.ocean_proximity.value_counts()

# For each we give a 1 for true and a 0 for false if the property is located in a specific proximity
dummies = pd.get_dummies(train_data.ocean_proximity)
dummies = dummies.astype(int)

# Add this back into the dataset
train_data = train_data.join(dummies)
train_data = train_data.drop('ocean_proximity', axis=1)
print(train_data.head())

# Redo the heatmap to see how the ocean proximity correlates with the target variable now that we can include ocean proximity
numeric_cols = train_data.select_dtypes(include=[float, int])
corr_matrix = numeric_cols.corr()

# Plot heatmap of the updated correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
plt.show()

# Visualize the change in price based on the proximity of the property to the ocean
plt.figure()
sns.scatterplot(x='latitude', y='longitude', data=train_data, hue='median_house_value', palette='coolwarm')
plt.show()

# Create a model using Linear Regression
from sklearn.linear_model import LinearRegression

X_train, y_train = train_data.drop(['median_house_value'], axis=1), train_data['median_house_value']
Regressor = LinearRegression()
Regressor.fit(X_train, y_train)

# Prepare the test data
test_data = X_test.join(y_test)
test_data['total_rooms'] = np.log(test_data['total_rooms']) + 1
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms']) + 1
test_data['population'] = np.log(test_data['population']) + 1
test_data['households'] = np.log(test_data['households']) + 1

# Convert categorical ocean proximity data to numeric using dummy variables
dummies = pd.get_dummies(test_data.ocean_proximity)
dummies = dummies.astype(int)

# Add the dummy variables to the test dataset
test_data = test_data.join(dummies)
test_data = test_data.drop('ocean_proximity', axis=1)

# Separate features and target variable in the test data
X_test, y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']
print(test_data.head())

# Make sure we did not drop any columns in the train data
train_data

# Evaluate the model