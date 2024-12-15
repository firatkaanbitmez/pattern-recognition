import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
from itertools import product

# Veri yolları
train_csv_path = r"C:\\Users\\FIRAT\\Desktop\\pattern-recognition\\Feature Extraction of Images-analysis\\Data\\mnist_train.csv"
test_csv_path = r"C:\\Users\\FIRAT\\Desktop\\pattern-recognition\\Feature Extraction of Images-analysis\\Data\\mnist_test.csv"

# Veri yükleme
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, 0].values
    images = df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
    return images, labels

train_images, train_labels = load_data(train_csv_path)
test_images, test_labels = load_data(test_csv_path)

# Özellik çıkarma fonksiyonları

def extract_glcm_features(image, distances=[1], angles=[0]):
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in properties:
        features.append(graycoprops(glcm, prop)[0, 0])
    return features

def extract_lbp_features(image, radius=3, n_points=24):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), 
                             bins=np.arange(0, n_points + 3), 
                             range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_run_length_features(image):
    # Simplified GLRLM extraction
    run_lengths = []
    for i in range(image.shape[0]):
        row = image[i, :]
        run_length = np.diff(np.where(np.diff(row) != 0)[0])
        run_lengths.extend(run_length)
    return [np.mean(run_lengths), np.std(run_lengths), shannon_entropy(image)]

def extract_lbglcm_features(image):
    lbp = local_binary_pattern(image, 24, 3, method='uniform')
    glcm = graycomatrix(lbp.astype(np.uint8), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return [contrast, energy, homogeneity]

def extract_features(images):
    features = []
    for img in images:
        glcm_f = extract_glcm_features(img)
        lbp_f = extract_lbp_features(img)
        glrlm_f = extract_run_length_features(img)
        lbglcm_f = extract_lbglcm_features(img)
        combined_features = np.hstack((glcm_f, lbp_f, glrlm_f, lbglcm_f))
        features.append(combined_features)
    return np.array(features)

# Özelliklerin çıkarılması
print("Extracting features for training data...")
train_features = extract_features(train_images)
print("Extracting features for test data...")
test_features = extract_features(test_images)

# Özelliklerin standardize edilmesi
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Veri bölünmesi
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# Modellerin tanımlanması
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Modellerin eğitilmesi ve değerlendirilmesi
results = {}
confusion_matrices = {}
roc_curves = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
    accuracy = accuracy_score(y_val, y_pred)
    results[name] = accuracy
    confusion_matrices[name] = confusion_matrix(y_val, y_pred)
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_val, y_proba[:, 1], pos_label=1)
        roc_curves[name] = (fpr, tpr)
    print(f"{name} Accuracy: {accuracy}")

# En iyi modeli seçme
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"Best Model: {best_model_name}")

# Test setinde en iyi modelin performansı
best_model.fit(train_features, train_labels)
test_predictions = best_model.predict(test_features)
print("Test Data Classification Report:")
print(classification_report(test_labels, test_predictions))

# Sonuçları görselleştirme
# Model başarılarının bar grafiği
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.title('Model Accuracy Comparison', fontsize=16)
plt.xlabel('Models', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.show()

# Confusion Matrix görselleştirme
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrices[best_model_name], annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
plt.title(f'Confusion Matrix for {best_model_name}', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.show()

# ROC AUC Curve
plt.figure(figsize=(10, 8))
for name, (fpr, tpr) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curves', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend()
plt.show()
