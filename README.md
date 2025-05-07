# Dry Bean Classification

This project focuses on classifying different types of dry beans using machine learning models. The dataset is sourced from the UCI Machine Learning Repository, and various models have been trained and evaluated to predict the type of dry bean based on several features.

## Project Structure

DryBean-Classification/
├── notebooks/                # Jupyter notebooks for data preprocessing and model training
│   ├── Preprocessing.ipynb   # Data preprocessing, cleaning, and feature engineering
│   ├── RandomForest.ipynb    # Random Forest classifier implementation
│   ├── SVM.ipynb             # Support Vector Machine classifier implementation
│   ├── XGBoost.ipynb         # XGBoost classifier implementation
│   ├── MLP.ipynb             # Multi-layer Perceptron classifier implementation
│   ├── KNN.ipynb             # K-Nearest Neighbors classifier implementation
│   ├── LogisticRegression.ipynb # Logistic Regression classifier implementation
│   └── Prediction.ipynb      # Model prediction and final evaluation
└── README.md                 # Project description and instructions


## Models Used

Several machine learning models were implemented and evaluated:

- **Random Forest**: A versatile ensemble method for classification.
- **Support Vector Machine (SVM)**: A robust model for classification tasks.
- **XGBoost**: An efficient and scalable gradient boosting model.
- **Multi-layer Perceptron (MLP)**: A neural network approach for classification.
- **K-Nearest Neighbors (KNN)**: A simple, yet effective algorithm for classification.
- **Logistic Regression**: A linear model for binary and multiclass classification.

## Model Evaluation

The performance of each model is evaluated using the following metrics:

- **Accuracy**: The proportion of correct predictions.
- **Precision & Recall**: Performance on positive class predictions.
- **F1 Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A detailed breakdown of prediction results.

## Tools & Technologies

- Python
- Jupyter Notebook
- **Libraries**:
  - Pandas & NumPy
  - Scikit-learn
  - Matplotlib & Seaborn
  - XGBoost

## Notes

- This project uses the [Dry Bean Dataset from UCI](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).
- The notebooks contain the full analysis pipeline, from data preprocessing to model training and evaluation.
- For further model comparisons and improvements, refer to the Prediction notebook.

---

Feel free to explore the notebooks for in-depth analysis and try different configurations to improve the classification performance.
