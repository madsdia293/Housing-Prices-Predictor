# Housing-Prices-Predictor

This project aims to predict housing prices in California using machine learning techniques. It involves exploring and preprocessing the data, engineering new features, training linear regression and random forest regression models, and evaluating their performance. The project is implemented in Python.

Link to the data set: https://www.kaggle.com/datasets/camnugent/california-housing-prices

## Overview

The project follows the following steps:

- Data Loading: The housing data is loaded from a CSV file using the pandas library.
- Data Preprocessing: Missing values are dropped from the dataset to ensure data integrity.
- Data Split: The data is split into training and testing sets using the train_test_split function from scikit-learn.
- Feature Engineering: Several features are engineered to improve the model's performance.
- Data Visualization: Histograms and heatmaps are created to visualize the distribution and correlations of the data.
- Model Training: Two models, Linear Regression and Random Forest Regression, are trained on the training data.
- Model Evaluation: The models are evaluated using the R-squared score on the testing data.
- Hyperparameter Tuning: Grid search is performed to find the best hyperparameters for the random forest model.
- Final Model Evaluation: The best random forest model is evaluated on the testing data.

## Engineered Features

The following features are engineered to improve the model's performance:

- Logarithmic Transformation: The columns 'total_rooms', 'total_bedrooms', 'population', and 'households' are transformed using the natural logarithm to achieve a more Gaussian-like distribution and reduce the impact of outliers.
- One-Hot Encoding: The categorical column 'ocean_proximity' is one-hot encoded using the pd.get_dummies function to represent each category as a binary feature.
- Additional Features: Two additional features, 'bedroom_ratio' and 'household_rooms', are created. 'bedroom_ratio' represents the ratio of total bedrooms to total rooms, indicating the proportion of bedrooms in each property. 'household_rooms' represents the average number of rooms per household.

## Models Created

The following models are created and evaluated:

- Linear Regression: A linear regression model is trained on the standardized training data using the LinearRegression class from scikit-learn. The model learns the linear relationship between the features and the target variable.
- Random Forest Regression: A random forest regression model is trained on the standardized training data using the RandomForestRegressor class from scikit-learn. The model learns the non-linear relationship between the features and the target variable using an ensemble of decision trees.
- Hyperparameter Tuning: Grid search is performed on the random forest model using the GridSearchCV class from scikit-learn to find the best combination of hyperparameters that minimizes the negative mean squared error. The best estimator is selected based on the grid search results.

## Requirements

To run this project, you will need the following:

Python 3.x: The project is implemented in Python, so make sure you have Python 3.x installed on your system.

Jupyter Notebook: Jupyter Notebook is used for executing the project code. Install Jupyter Notebook by running the following command in your terminal:
    pip install jupyter

Required Python libraries: The project relies on various Python libraries such as pandas, scikit-learn, and matplotlib. Install these libraries by running the following command:
    pip install pandas numpy scikit-learn matplotlib seaborn

Housing dataset: The project requires a housing dataset in CSV format. You can obtain the dataset from:
    https://www.kaggle.com/datasets/camnugent/california-housing-prices
    
Once you have fulfilled these requirements, you can proceed to execute the project code and follow the steps outlined in the project overview section of this README file.
