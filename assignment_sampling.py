# -*- coding: utf-8 -*-
"""Assignment_Sampling.ipynb

Original file is located at
    https://colab.research.google.com/drive/1QNc1oXjt_t8HA66yqkZBRm3RVSs43qrp
"""

import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Step 1: Load the dataset
data = pd.read_csv("Creditcard_data.csv")

# Step 2: Balance the dataset
# Assuming the target column is 'Class', adjust as needed
majority = data[data['Class'] == 0]
minority = data[data['Class'] == 1]

# Oversample minority class
minority_oversampled = resample(minority,
                                replace=True,
                                n_samples=len(majority),
                                random_state=42)

balanced_data = pd.concat([majority, minority_oversampled])

# Step 3: Create five samples
sample_sizes = [len(balanced_data) // 5] * 5  # Equal sizes
samples = [balanced_data.sample(size, random_state=i) for i, size in enumerate(sample_sizes)]

# Step 4: Sampling techniques (Random, Stratified, etc.)
techniques = {
    "Sampling1": lambda df: df.sample(frac=0.8, random_state=1),
    "Sampling2": lambda df: df.sample(frac=0.8, random_state=2),
    "Sampling3": lambda df: df.sample(frac=0.8, random_state=3),
    "Sampling4": lambda df: df.sample(frac=0.8, random_state=4),
    "Sampling5": lambda df: df.sample(frac=0.8, random_state=5),
}
sampled_data = {name: func(balanced_data) for name, func in techniques.items()}

# Step 5: Machine learning models
models = {
    "M1": LogisticRegression(),
    "M2": DecisionTreeClassifier(),
    "M3": RandomForestClassifier(),
    "M4": SVC(),
    "M5": KNeighborsClassifier(),
}

# Preparing results storage
results = pd.DataFrame(columns=["Model", "Sampling Technique", "Accuracy"])

# Train and evaluate models
for sample_name, sample_df in sampled_data.items():
    X = sample_df.drop('Class', axis=1)
    y = sample_df['Class']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Store results
        results = pd.concat([results, pd.DataFrame({
            "Model": [model_name],
            "Sampling Technique": [sample_name],
            "Accuracy": [accuracy]
        })])

# Display results
results.reset_index(drop=True, inplace=True)
print(results)

# Save results to a CSV file
results.to_csv('sampling_model_results.csv', index=False)
