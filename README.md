# Overview
Detecting credit card fraud is a challenging task due to the heavily imbalanced nature of transaction datasets, which can negatively affect machine learning model performance. This project focuses on:

1. Addressing class imbalance by applying the Synthetic Minority Oversampling Technique (SMOTE).
2. Generating five samples of varying sizes using different sampling methods.
3. Training and evaluating five machine learning algorithms on these sampled datasets.
4. Analyzing and comparing the models’ performance to determine the most effective sampling technique for each.

# Dataset
The dataset used, Creditcard_data.csv (provided for the project), includes the following details:

1. Features (V1 to V28): Derived from the principal components of transaction data.
2. Time: Indicates the seconds elapsed between consecutive transactions.
3. Amount: Reflects the transaction value.
4. Class: Fraud indicator, where 0 represents legitimate transactions and 1 indicates fraud.

## Class Imbalance
The dataset is highly skewed, with the following class distribution:
1. Class 0 (Legitimate Transactions): Comprising 98.83% of the data.
2. Class 1 (Fraudulent Transactions): Making up only 1.17% of the data.

# Sampling Techniques
To overcome class imbalance, various sampling methods were employed:
Random Sampling: Randomly selects a subset of records.
Stratified Sampling: Ensures the class proportions in the dataset are preserved.
Systematic Sampling: Picks every nth record systematically.
Cluster Sampling: Divides the dataset into clusters and selects entire clusters for analysis.
Oversampling: Increases the representation of the minority class to achieve balance.
# Machine Learning Models
The following algorithms were trained on the sampled datasets:
1.Logistic Regression
2.Decision Tree Classifier
3.Random Forest Classifier
4.Support Vector Machine (SVM)
5.K-Nearest Neighbors

# Observations
Logistic Regression: Showed optimal performance with Stratified Sampling and Oversampling.

Decision Tree: Delivered the best results when trained on data generated through Stratified Sampling.

Random Forest: Achieved its highest accuracy with Oversampling techniques.

SVM: Performed most effectively on oversampled datasets.

Gradient Boosting: Worked best with Stratified Sampling.
