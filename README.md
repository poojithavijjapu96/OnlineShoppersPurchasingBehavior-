# Online Shoppers Purchasing Behavior

<h3>Introduction and Goal:</h3>
Welcome to the Online Shoppers Purchasing Behavior project, where we analyze a dataset containing various predictor variables related to customer visits to an e-commerce website. Our goal is to identify revenue-generating customers and target them effectively to optimize marketing strategies and enhance sales performance.

<h3>Dataset:</h3>
The data is a collection of various predictor variables related to a session of a customer's visit to an e-commerce website. Some features of the data : 

_<h4>1. Multivariate: </h4>_ The dataset is multi-variate, consisting of a combination of numerical and categorical features.
_<h4>2. Binary Response Variable:</h4>_ The response variable indicates whether a customer generated revenue during their visit, with True or False values.
_<h4>3. Imbalanced Data:</h4>_ The dataset exhibits class imbalance, with a significantly lower proportion of customers generating revenue compared to those who do not.

<h3>Data Analysis:</h3>
Our primary objective is to identify revenue-generating customers and develop a predictive model that maximizes the identification of true positives. To address the imbalance in the data, we perform oversampling. We explore various regression models, including logistic regression, random forest, and lasso regression, aiming to achieve high sensitivity (true positive rates). Our analysis reveals that the lasso regression model demonstrates the highest sensitivity at 88.3%.

<h3>Highlights:</h3>

_<h4>1. Oversampling of Data:</h4>_ We employ oversampling techniques to address the class imbalance in the dataset.
_<h4>2. Multiple Regression Models:</h4>_ We explore and compare multiple regression models to identify the best-performing model for predicting revenue-generating customers.

<h3>Report and Code:<h3>

The detailed report for this analysis is available in the file named "Online Shoppers Purchasing Behavior.pdf".
The code for data analysis and model development is provided in two Python script files: "OnlineShoppersPurchasingBehavior_code_Part1.py" and "OnlineShoppersPurchasingBehavior_code_Part2.py".
