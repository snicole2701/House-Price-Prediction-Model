# House-Price-Prediction-Model
## Objective: Develop a predictive model using Python to estimate housing prices based on various features.

Data Source: The dataset consists of various features affecting house prices, such as the number of rooms, the population, households, and proximity to the ocean.

## Steps Taken:

### Data Loading and Cleaning:

Loaded the dataset into a pandas DataFrame.

Identified and dropped rows with null values to ensure data integrity.

### Data Splitting:

Split the data into training and testing sets using train_test_split from sklearn, with 75% for training and 25% for testing.

### Exploratory Data Analysis (EDA):

Generated correlation matrices to understand relationships between features.

Visualized the correlations using heatmaps created with seaborn.

### Feature Transformation:

Applied logarithmic transformation to features like total_rooms, total_bedrooms, population, and households to handle skewed distributions.

Created dummy variables for the categorical feature ocean_proximity to include it in the model.

### Data Visualization:

Created scatter plots, heat maps and histograms to visualize the data.

### Model Building:

Built a Linear Regression model using sklearn.

Trained the model on the training data and evaluated it on the test data.

### Results:

The model achieved an R-squared value of 0.669, indicating that approximately 66.9% of the variability in house prices can be explained by the features included in the model.

### Technologies Used:

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

### Future Work:

Explore additional features and more advanced machine learning models to improve prediction accuracy.

Implement feature engineering techniques to better capture the relationships within the data.
