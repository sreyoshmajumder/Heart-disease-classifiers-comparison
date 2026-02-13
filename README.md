# Heart-disease-classifiers-comparison
Comprehensive comparison of 20+ supervised learning models on the Kaggle Heart Disease UCI dataset using scikit-learn and modern boosting libraries.
Heart Disease ML Benchmark

This project compares multiple supervised learning algorithms on the Kaggle Heart Disease UCI dataset to evaluate their performance on a real-world binary classification problem.
📌 Project Overview

    Task: Predict whether a patient has heart disease (binary classification).

    Dataset: Heart Disease UCI (Kaggle).

    Models: 20+ classic and ensemble classifiers (linear models, trees, boosting, SVMs, Naive Bayes, KNN, discriminant analysis, neural network).

    Output: Metrics table, visualizations, and confusion matrices for top models.

🧠 Algorithms Implemented

    Logistic Regression

    Ridge Classifier

    SGD Classifier

    Decision Tree

    Random Forest

    Extra Trees

    AdaBoost

    Gradient Boosting

    XGBoost

    LightGBM

    CatBoost

    SVM (RBF, Linear, LinearSVC)

    Gaussian Naive Bayes

    Bernoulli Naive Bayes

    K-Nearest Neighbors

    Linear Discriminant Analysis (LDA)

    Quadratic Discriminant Analysis (QDA)

    Multi-Layer Perceptron (MLP)

📂 Dataset

    Source: Kaggle – Heart Disease UCI dataset

    File used: heart.csv

    Target column: target (1 = disease, 0 = no disease)

Download the dataset from Kaggle and place heart.csv in the project root folder (same directory as the main Python script).
🚀 Getting Started
1. Clone the repository

bash
git clone https://github.com/your-username/heart-disease-ml-benchmark.git
cd heart-disease-ml-benchmark

2. Create and activate a virtual environment (optional but recommended)

bash
python -m venv venv
venv\Scripts\activate       # Windows
# or
source venv/bin/activate   # macOS/Linux

3. Install dependencies

bash
pip install -r requirements.txt

4. Add the dataset

    Download heart.csv from Kaggle.

    Place it in the project root (same folder as supervised_ml_comparison.py).

5. Run the benchmark

bash
python supervised_ml_comparison.py

📊 Outputs

The script generates:

    model_comparison_results.csv – metrics for all models (accuracy, precision, recall, F1, ROC-AUC, CV mean/std, training time).

    model_comparison_visualizations.png – comparison plots (accuracy, F1, ROC-AUC, training time, CV scores, etc.).

    top3_confusion_matrices.png – confusion matrices for the top 3 models.

These files help quickly see which models perform best and how they trade off speed vs accuracy.
📈 Evaluation Metrics

For each model, the script computes:

    Accuracy

    Precision

    Recall

    F1-Score

    ROC-AUC (when possible)

    5-fold cross-validation mean and standard deviation

    Training time (seconds)

🧪 Tech Stack

    Python

    scikit-learn

    XGBoost

    LightGBM

    CatBoost

    pandas, NumPy

    matplotlib, seaborn

🔍 Possible Extensions

    Hyperparameter tuning for top models

    Feature importance analysis

    Comparison with other datasets (e.g., Titanic, Iris, etc.)

    Adding regression benchmarks in a similar framework
