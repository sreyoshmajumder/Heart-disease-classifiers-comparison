# ============================================================================
# COMPREHENSIVE SUPERVISED LEARNING MODEL COMPARISON
# Dataset: Heart Disease UCI (Download from Kaggle)
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)

# ============================================================================
# IMPORTING ALL SUPERVISED LEARNING MODELS
# ============================================================================

# Linear Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier

# Tree-Based Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                              AdaBoostClassifier, GradientBoostingClassifier)

# Support Vector Machines
from sklearn.svm import SVC, LinearSVC

# Naive Bayes
from sklearn.naive_bayes import GaussianNB, BernoulliNB

# Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Neural Networks
from sklearn.neural_network import MLPClassifier

# Advanced Ensemble Methods
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import time

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================

print("="*80)
print("COMPREHENSIVE SUPERVISED LEARNING MODEL COMPARISON")
print("="*80)

# Load dataset (Make sure 'heart.csv' is in the same directory)
# Download from: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
df = pd.read_csv('heart.csv')

print("\n📊 DATASET OVERVIEW")
print("-"*80)
print(f"Dataset Shape: {df.shape}")
print(f"Features: {df.shape[1] - 1}")
print(f"Samples: {df.shape[0]}")
print("\nFirst 5 rows:")
print(df.head())

print("\n📈 DATASET INFO:")
print(df.info())

print("\n📉 STATISTICAL SUMMARY:")
print(df.describe())

print("\n🎯 TARGET DISTRIBUTION:")
print(df['target'].value_counts())
print(f"Class Balance: {df['target'].value_counts(normalize=True).to_dict()}")

print("\n🔍 MISSING VALUES:")
print(df.isnull().sum())

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Train set: {X_train.shape}")
print(f"✅ Test set: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Feature scaling completed")

# ============================================================================
# STEP 3: DEFINE ALL MODELS
# ============================================================================

print("\n" + "="*80)
print("INITIALIZING ALL SUPERVISED LEARNING MODELS")
print("="*80)

models = {
    # Linear Models
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Ridge Classifier': RidgeClassifier(random_state=42),
    'SGD Classifier': SGDClassifier(random_state=42, max_iter=1000),
    
    # Tree-Based Models
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    
    # Ensemble Boosting
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
    
    # Support Vector Machines
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
    'Linear SVC': LinearSVC(random_state=42, max_iter=1000),
    
    # Naive Bayes
    'Gaussian Naive Bayes': GaussianNB(),
    'Bernoulli Naive Bayes': BernoulliNB(),
    
    # Nearest Neighbors
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    
    # Discriminant Analysis
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
    
    # Neural Networks
        'Multi-Layer Perceptron': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

}

print(f"✅ Total models to compare: {len(models)}")
for i, model_name in enumerate(models.keys(), 1):
    print(f"  {i}. {model_name}")

# ============================================================================
# STEP 4: TRAIN AND EVALUATE ALL MODELS
# ============================================================================

print("\n" + "="*80)
print("TRAINING AND EVALUATING ALL MODELS")
print("="*80)

results = []

for name, model in models.items():
    print(f"\n🔄 Training: {name}...")
    
    start_time = time.time()
    
    try:
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Get prediction probabilities (if available)
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test_scaled)
        else:
            y_pred_proba = y_pred
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = np.nan
        
        training_time = time.time() - start_time
        
        # Store results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'CV Mean': cv_scores.mean(),
            'CV Std': cv_scores.std(),
            'Training Time (s)': training_time
        })
        
        print(f"  ✅ Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f} | Time: {training_time:.2f}s")
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        continue

# ============================================================================
# STEP 5: CREATE RESULTS DATAFRAME AND SORT
# ============================================================================

print("\n" + "="*80)
print("FINAL RESULTS - ALL MODELS COMPARISON")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
results_df['Rank'] = results_df.index + 1

# Display results
print("\n" + results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('model_comparison_results.csv', index=False)
print("\n✅ Results saved to 'model_comparison_results.csv'")

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Create subplot figure
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Comprehensive Model Comparison - Heart Disease Prediction', fontsize=20, fontweight='bold')

# 1. Accuracy Comparison (Bar Plot)
ax1 = axes[0, 0]
top_10 = results_df.head(10)
colors = sns.color_palette("viridis", len(top_10))
ax1.barh(top_10['Model'], top_10['Accuracy'], color=colors)
ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Top 10 Models by Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlim([0.5, 1.0])
for i, v in enumerate(top_10['Accuracy']):
    ax1.text(v + 0.01, i, f'{v:.4f}', va='center')

# 2. Precision vs Recall
ax2 = axes[0, 1]
scatter = ax2.scatter(results_df['Precision'], results_df['Recall'], 
                     c=results_df['Accuracy'], cmap='RdYlGn', s=200, alpha=0.7)
ax2.set_xlabel('Precision', fontsize=12, fontweight='bold')
ax2.set_ylabel('Recall', fontsize=12, fontweight='bold')
ax2.set_title('Precision vs Recall (colored by Accuracy)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax2, label='Accuracy')

# Add model names
for idx, row in results_df.iterrows():
    if idx < 10:  # Top 10 only to avoid clutter
        ax2.annotate(row['Model'], (row['Precision'], row['Recall']), 
                    fontsize=8, alpha=0.7)

# 3. F1-Score Comparison
ax3 = axes[0, 2]
top_10_f1 = results_df.nlargest(10, 'F1-Score')
ax3.barh(top_10_f1['Model'], top_10_f1['F1-Score'], color='coral')
ax3.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
ax3.set_title('Top 10 Models by F1-Score', fontsize=14, fontweight='bold')
ax3.set_xlim([0.5, 1.0])
for i, v in enumerate(top_10_f1['F1-Score']):
    ax3.text(v + 0.01, i, f'{v:.4f}', va='center')

# 4. Training Time Comparison
ax4 = axes[1, 0]
sorted_by_time = results_df.sort_values('Training Time (s)').head(10)
ax4.barh(sorted_by_time['Model'], sorted_by_time['Training Time (s)'], color='skyblue')
ax4.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax4.set_title('Fastest 10 Models by Training Time', fontsize=14, fontweight='bold')

# 5. Cross-Validation Score with Error Bars
ax5 = axes[1, 1]
top_10_cv = results_df.nlargest(10, 'CV Mean')
ax5.barh(top_10_cv['Model'], top_10_cv['CV Mean'], 
         xerr=top_10_cv['CV Std'], color='lightgreen', capsize=5)
ax5.set_xlabel('CV Mean Accuracy', fontsize=12, fontweight='bold')
ax5.set_title('Top 10 Models by Cross-Validation Score', fontsize=14, fontweight='bold')
ax5.set_xlim([0.5, 1.0])

# 6. ROC-AUC Comparison
ax6 = axes[1, 2]
roc_data = results_df.dropna(subset=['ROC-AUC']).nlargest(10, 'ROC-AUC')
ax6.barh(roc_data['Model'], roc_data['ROC-AUC'], color='purple', alpha=0.7)
ax6.set_xlabel('ROC-AUC Score', fontsize=12, fontweight='bold')
ax6.set_title('Top 10 Models by ROC-AUC', fontsize=14, fontweight='bold')
ax6.set_xlim([0.5, 1.0])
for i, v in enumerate(roc_data['ROC-AUC']):
    ax6.text(v + 0.01, i, f'{v:.4f}', va='center')

plt.tight_layout()
plt.savefig('model_comparison_visualizations.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved as 'model_comparison_visualizations.png'")
plt.show()

# ============================================================================
# STEP 7: DETAILED ANALYSIS OF TOP 3 MODELS
# ============================================================================

print("\n" + "="*80)
print("DETAILED ANALYSIS - TOP 3 MODELS")
print("="*80)

top_3_models = results_df.head(3)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices - Top 3 Models', fontsize=16, fontweight='bold')

for idx, (ax, row) in enumerate(zip(axes, top_3_models.itertuples())):
    model_name = row.Model
    model = models[model_name]
    
    # Re-train and predict
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                cbar_kws={'label': 'Count'})
    ax.set_title(f'{model_name}\nAccuracy: {row.Accuracy:.4f}', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
    
    print(f"\n🏆 RANK {idx + 1}: {model_name}")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

plt.tight_layout()
plt.savefig('top3_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("\n✅ Confusion matrices saved as 'top3_confusion_matrices.png'")
plt.show()

# ============================================================================
# STEP 8: SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL SUMMARY AND RECOMMENDATIONS")
print("="*80)

best_model = results_df.iloc[0]
print(f"\n🥇 BEST OVERALL MODEL: {best_model['Model']}")
print(f"   ✓ Accuracy: {best_model['Accuracy']:.4f}")
print(f"   ✓ F1-Score: {best_model['F1-Score']:.4f}")
print(f"   ✓ ROC-AUC: {best_model['ROC-AUC']:.4f}")
print(f"   ✓ Training Time: {best_model['Training Time (s)']:.2f}s")

fastest_model = results_df.nsmallest(1, 'Training Time (s)').iloc[0]
print(f"\n⚡ FASTEST MODEL: {fastest_model['Model']}")
print(f"   ✓ Training Time: {fastest_model['Training Time (s)']:.4f}s")
print(f"   ✓ Accuracy: {fastest_model['Accuracy']:.4f}")

best_cv_model = results_df.nlargest(1, 'CV Mean').iloc[0]
print(f"\n🎯 MOST STABLE MODEL (Cross-Validation): {best_cv_model['Model']}")
print(f"   ✓ CV Mean: {best_cv_model['CV Mean']:.4f} ± {best_cv_model['CV Std']:.4f}")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  1. model_comparison_results.csv")
print("  2. model_comparison_visualizations.png")
print("  3. top3_confusion_matrices.png")
