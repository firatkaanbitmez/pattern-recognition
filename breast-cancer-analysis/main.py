
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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

# Load the dataset
df = pd.read_csv("wdbc.data", header=None, names=column_names)

# Step 2: Preprocess the data
# Encode the 'Diagnosis' column
encoder = LabelEncoder()
df['Diagnosis'] = encoder.fit_transform(df['Diagnosis'])  # M -> 1, B -> 0

# Drop the 'ID' column as it's not informative for classification
df = df.drop(columns=['ID'])

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(columns=['Diagnosis']))

# Create a DataFrame with scaled features
feature_names = df.columns.drop('Diagnosis')
df_scaled = pd.DataFrame(scaled_features, columns=feature_names)
df_scaled['Diagnosis'] = df['Diagnosis']

# Step 3: Dimensionality Reduction
# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
df_scaled['PCA1'] = pca_result[:, 0]
df_scaled['PCA2'] = pca_result[:, 1]

# LDA
lda = LDA(n_components=1)
lda_result = lda.fit_transform(scaled_features, df_scaled['Diagnosis'])
df_scaled['LDA1'] = lda_result[:, 0]

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(scaled_features)
df_scaled['tSNE1'] = tsne_result[:, 0]
df_scaled['tSNE2'] = tsne_result[:, 1]

# Step 4: Train-Test Split
X = df_scaled.drop(columns=['Diagnosis'])
y = df_scaled['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Classifier and Evaluate
# Using Random Forest as an example
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Step 6: Visualizations
# PCA Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Diagnosis', data=df_scaled, palette='viridis')
plt.title("PCA Visualization")
plt.savefig("PCA_Visualization.png")

# t-SNE Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x='tSNE1', y='tSNE2', hue='Diagnosis', data=df_scaled, palette='coolwarm')
plt.title("t-SNE Visualization")
plt.savefig("tSNE_Visualization.png")
