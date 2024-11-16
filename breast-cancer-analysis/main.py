import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib

# ===================== Sabit Değerler =====================
RANDOM_STATE = 42

# ===================== Dinamik Klasör Yolları =====================
base_dir = os.path.dirname(os.path.abspath(__file__))  # Kodun bulunduğu klasör
data_dir = os.path.join(base_dir, "Data")  # Veri klasörü
output_dir = os.path.join(base_dir, "Output")  # Çıktılar için klasör
models_dir = os.path.join(output_dir, "Models")  # Modeller için klasör
visuals_dir = os.path.join(output_dir, "Visuals")  # Görseller için klasör

# Gerekli klasörleri oluştur
for directory in [output_dir, models_dir, visuals_dir]:
    os.makedirs(directory, exist_ok=True)

# ===================== Veri Yükleme =====================
data_path = os.path.join(data_dir, "wdbc.data")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_path}")

# Kolon isimleri
column_names = [
    "ID",
    "Diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

# Veri setini yükle
df = pd.read_csv(data_path, header=None, names=column_names)

# ===================== Veri Ön İşleme =====================
# Diagnosis sütunu (M: 1, B: 0)
df["Diagnosis"] = LabelEncoder().fit_transform(df["Diagnosis"])
df.drop(columns=["ID"], inplace=True)  # ID sütunu kaldırıldı

# Özellikler ve hedef değişken ayrımı
features = df.drop(columns=["Diagnosis"])
target = df["Diagnosis"]

# Veriyi ölçeklendirme
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))  # Scaler kaydedildi

# Eğitim ve test verisi
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, target, test_size=0.2, random_state=RANDOM_STATE
)

# ===================== Boyut İndirgeme Fonksiyonu =====================
def apply_dimensionality_reduction(method, n_components, X, y=None):
    """
    Boyut indirgeme işlemini gerçekleştirir.
    Args:
        method (str): PCA, LDA veya t-SNE
        n_components (int): Hedef boyut sayısı
        X (array): Özellik matrisi
        y (array, optional): Hedef değişken (LDA için gerekli)
    Returns:
        array: Boyutları indirgenmiş veri
    """
    if method == "PCA":
        reducer = PCA(n_components=n_components, random_state=RANDOM_STATE)
    elif method == "LDA":
        reducer = LDA(n_components=n_components)
    elif method == "t-SNE":
        reducer = TSNE(n_components=n_components, random_state=RANDOM_STATE)
    return reducer.fit_transform(X, y) if y is not None else reducer.fit_transform(X)

# ===================== Boyut İndirgeme =====================
# PCA
pca_features = apply_dimensionality_reduction("PCA", 2, scaled_features)
# LDA
lda_features = apply_dimensionality_reduction("LDA", 1, scaled_features, target)
# t-SNE
tsne_features = apply_dimensionality_reduction("t-SNE", 2, scaled_features)

# ===================== Model Tanımları =====================
models = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
}

# ===================== Model Değerlendirme =====================
def evaluate_models(X, y, models, method_name):
    """
    Modellerin performansını değerlendirir.
    Args:
        X (array): Özellik matrisi
        y (array): Hedef değişken
        models (dict): Model isimleri ve nesneleri
        method_name (str): Kullanılan boyut indirgeme yöntemi
    Returns:
        dict: Model başına doğruluk skorları
    """
    results = {}
    kfold = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    for model_name, model in models.items():
        scores = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
        results[model_name] = scores.mean()
        print(f"{method_name} - {model_name}: Ortalama Başarı: {scores.mean():.4f}")
    return results

# Değerlendirme
results_summary = {
    "No Reduction": evaluate_models(scaled_features, target, models, "No Reduction"),
    "PCA": evaluate_models(pca_features, target, models, "PCA"),
    "LDA": evaluate_models(lda_features, target, models, "LDA"),
    "t-SNE": evaluate_models(tsne_features, target, models, "t-SNE"),
}

# ===================== Sonuçların Kaydedilmesi =====================
results_text_path = os.path.join(output_dir, "results_summary.txt")
with open(results_text_path, "w") as file:
    for method, results in results_summary.items():
        file.write(f"==== {method} ====\n")
        for model, accuracy in results.items():
            file.write(f"{model}: {accuracy:.4f}\n")
        file.write("\n")

# ===================== Görselleştirme =====================
def plot_reduction_scatter(features, target, title, save_path):
    plt.figure(figsize=(10, 8))
    if features.shape[1] == 2:
        plt.scatter(features[:, 0], features[:, 1], c=target, cmap="viridis", edgecolor="k")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
    elif features.shape[1] == 1:
        plt.scatter(features[:, 0], np.zeros_like(features[:, 0]), c=target, cmap="viridis", edgecolor="k")
        plt.xlabel("Component 1")
        plt.ylabel("Class Projection")
    plt.title(title)
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()

plot_reduction_scatter(pca_features, target, "PCA Scatter Plot", os.path.join(visuals_dir, "PCA_Scatter.png"))
plot_reduction_scatter(lda_features, target, "LDA Scatter Plot", os.path.join(visuals_dir, "LDA_Scatter.png"))
plot_reduction_scatter(tsne_features, target, "t-SNE Scatter Plot", os.path.join(visuals_dir, "tSNE_Scatter.png"))

print("Tüm işlemler tamamlandı. Çıktılar Output klasörüne kaydedildi.")
