
# Comprehensive Breast Cancer Classification Project

# Importing required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Load the dataset
column_names = [
    "ID", "Diagnosis", 
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", 
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "Data", "wdbc.data")
output_dir = os.path.join(base_dir, "Output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


if not os.path.exists(data_path):
    print(f"Dosya bulunamadı: {data_path}")
else:
    df = pd.read_csv(data_path, header=None, names=column_names)
    
# Step 2: Data cleaning and preprocessing
# Encode the 'Diagnosis' column
encoder = LabelEncoder()
df['Diagnosis'] = encoder.fit_transform(df['Diagnosis'])  # M -> 1, B -> 0

# Drop the 'ID' column as it's not useful for classification
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# Check for missing values and handle them
df = df.dropna()

# Standardize the features
scaler = StandardScaler()
features = df.drop(columns=['Diagnosis'])
scaled_features = scaler.fit_transform(features)

# Step 3: Dimensionality Reduction
def apply_dimensionality_reduction(method, n_components, X, y=None):
    if method == "PCA":
        model = PCA(n_components=n_components)
    elif method == "LDA":
        model = LDA(n_components=n_components)
    elif method == "t-SNE":
        model = TSNE(n_components=n_components, random_state=42)
    return model.fit_transform(X, y) if y is not None else model.fit_transform(X)

# Applying dimensionality reduction
pca_features = apply_dimensionality_reduction("PCA", 2, scaled_features)
lda_features = apply_dimensionality_reduction("LDA", 1, scaled_features, df['Diagnosis'])
tsne_features = apply_dimensionality_reduction("t-SNE", 2, scaled_features)

# Step 4: Classification Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(kernel='linear', probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Helper function for training and evaluation
def evaluate_models(X, y, models, method_name):
    results = {}
    kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for model_name, model in models.items():
        scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
        results[model_name] = scores
        print(f"{model_name} with {method_name}: Mean Accuracy = {scores.mean():.2f}")
    return results

# Evaluating models for each scenario
results_summary = {}
results_summary["No Reduction"] = evaluate_models(scaled_features, df['Diagnosis'], models, "No Reduction")
results_summary["PCA"] = evaluate_models(pca_features, df['Diagnosis'], models, "PCA")
results_summary["LDA"] = evaluate_models(lda_features, df['Diagnosis'], models, "LDA")
results_summary["t-SNE"] = evaluate_models(tsne_features, df['Diagnosis'], models, "t-SNE")

# Visualizations
# PCA Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=df['Diagnosis'], palette='viridis')
plt.title("PCA Visualization")
pca_output_path = os.path.join(output_dir, "PCA_Visualization.png")
plt.savefig(pca_output_path)


# t-SNE Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=df['Diagnosis'], palette='coolwarm')
plt.title("t-SNE Visualization")
tsne_output_path = os.path.join(output_dir, "tSNE_Visualization.png")
plt.savefig(tsne_output_path)