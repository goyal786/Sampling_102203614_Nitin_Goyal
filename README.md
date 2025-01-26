# Overview
Detecting credit card fraud is a challenging task due to the heavily imbalanced nature of transaction datasets, which can negatively affect machine learning model performance. This project focuses on:

# Addressing class imbalance by applying the Synthetic Minority Oversampling Technique (SMOTE).
Generating five samples of varying sizes using different sampling methods.
Training and evaluating five machine learning algorithms on these sampled datasets.
Analyzing and comparing the models’ performance to determine the most effective sampling technique for each.
## Dataset
The dataset used, Creditcard_data.csv (provided for the project), includes the following details:

# Features (V1 to V28): Derived from the principal components of transaction data.
Time: Indicates the seconds elapsed between consecutive transactions.
Amount: Reflects the transaction value.
Class: Fraud indicator, where 0 represents legitimate transactions and 1 indicates fraud.
## Class Imbalance
The dataset is highly skewed, with the following class distribution:

Class 0 (Legitimate Transactions): Comprising 98.83% of the data.
Class 1 (Fraudulent Transactions): Making up only 1.17% of the data.
# Sampling Techniques
To overcome class imbalance, various sampling methods were employed:

Random Sampling: Randomly selects a subset of records.
Stratified Sampling: Ensures the class proportions in the dataset are preserved.
Systematic Sampling: Picks every nth record systematically.
Cluster Sampling: Divides the dataset into clusters and selects entire clusters for analysis.
Oversampling: Increases the representation of the minority class to achieve balance.
# Machine Learning Models
The following algorithms were trained on the sampled datasets:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Support Vector Machine (SVM)

Gradient Boosting

# Observations
Logistic Regression: Showed optimal performance with Stratified Sampling and Oversampling.
Decision Tree: Delivered the best results when trained on data generated through Stratified Sampling.
Random Forest: Achieved its highest accuracy with Oversampling techniques.
SVM: Performed most effectively on oversampled datasets.
Gradient Boosting: Worked best with Stratified Sampling.
