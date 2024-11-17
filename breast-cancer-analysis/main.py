import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
from fpdf import FPDF
import seaborn as sns

# ===================== Sabit Değerler =====================
RANDOM_STATE = 42
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wdbc/wdbc.data"

# ===================== Dinamik Klasör Yolları =====================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "Data")
output_dir = os.path.join(base_dir, "Output")
models_dir = os.path.join(output_dir, "Models")
visuals_dir = os.path.join(output_dir, "Visuals")
pca_visuals_dir = os.path.join(visuals_dir, "PCA")
lda_visuals_dir = os.path.join(visuals_dir, "LDA")
tsne_visuals_dir = os.path.join(visuals_dir, "t-SNE")
noreduction_visuals_dir = os.path.join(visuals_dir, "NoReduction")
comparison_visuals_dir = os.path.join(visuals_dir, "Comparison")
report_dir = os.path.join(output_dir, "Reports")

# Gerekli klasörleri oluştur
for directory in [data_dir, output_dir, models_dir, visuals_dir, pca_visuals_dir, lda_visuals_dir, tsne_visuals_dir, noreduction_visuals_dir, comparison_visuals_dir, report_dir]:
    os.makedirs(directory, exist_ok=True)

# ===================== Veri Yükleme =====================
data_path = os.path.join(data_dir, "wdbc.data")
if not os.path.exists(data_path):
    print(f"Veri dosyası bulunamadı. '{data_path}' yoluna indiriliyor...")
    df_data = pd.read_csv(DATA_URL, header=None)
    df_data.to_csv(data_path, header=False, index=False)
else:
    df_data = pd.read_csv(data_path, header=None)

# Kolon isimleri
column_names = [
    "ID", "Diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave_points_worst", "symmetry_worst",
    "fractal_dimension_worst",
]
df = pd.read_csv(data_path, header=None, names=column_names)

# ===================== Veri Ön İşleme =====================
df["Diagnosis"] = LabelEncoder().fit_transform(df["Diagnosis"])  # M: 1, B: 0
df.drop(columns=["ID"], inplace=True)  # ID sütunu kaldırıldı

# Özellikler ve hedef değişken ayrımı
features = df.drop(columns=["Diagnosis"])
target = df["Diagnosis"]

# Veriyi ölçeklendirme
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

# Eğitim ve test verisi
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, target, test_size=0.2, random_state=RANDOM_STATE, stratify=target
)

# ===================== PCA Varyans Analizi =====================
pca = PCA().fit(scaled_features)
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('Açıklanan Varyans Oranı')
plt.title('PCA - Açıklanan Varyans Oranı')
plt.grid()
plt.savefig(os.path.join(pca_visuals_dir, "PCA_Variance_Explained.png"))
plt.close()

# ===================== Boyut İndirgeme Fonksiyonu =====================
def apply_dimensionality_reduction(method, n_components, X_train, X_test, y_train=None):
    if method == "PCA":
        reducer = PCA(n_components=n_components, random_state=RANDOM_STATE)
    elif method == "LDA":
        reducer = LDA(n_components=min(n_components, y_train.nunique() - 1))
    elif method == "t-SNE":
        reducer = TSNE(n_components=n_components, perplexity=30, learning_rate=200, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Desteklenmeyen yöntem: {method}")
    
    X_train_reduced = reducer.fit_transform(X_train, y_train) if y_train is not None else reducer.fit_transform(X_train)
    X_test_reduced = reducer.transform(X_test) if hasattr(reducer, "transform") else reducer.fit_transform(X_test)
    return X_train_reduced, X_test_reduced

# ===================== Boyut İndirgeme =====================
# PCA
X_train_pca, X_test_pca = apply_dimensionality_reduction("PCA", 2, X_train, X_test)
# LDA
X_train_lda, X_test_lda = apply_dimensionality_reduction("LDA", 1, X_train, X_test, y_train)
# t-SNE
X_train_tsne, X_test_tsne = apply_dimensionality_reduction("t-SNE", 2, X_train, X_test)

# ===================== Görselleştirme Fonksiyonları =====================
def plot_scatter(features, target, title, filename, directory):
    plt.figure(figsize=(10, 8))
    plt.scatter(features[:, 0], features[:, 1], c=target, cmap="viridis", edgecolor="k", alpha=0.7)
    plt.colorbar(label="Sınıf")
    plt.title(title)
    plt.xlabel("Bileşen 1")
    plt.ylabel("Bileşen 2")
    plt.savefig(os.path.join(directory, filename))
    plt.close()

# Görselleştirme
plot_scatter(X_train_pca, y_train, "PCA - 2 Bileşen", "PCA_Scatter.png", pca_visuals_dir)
plot_scatter(X_train_tsne, y_train, "t-SNE - 2 Bileşen", "tSNE_Scatter.png", tsne_visuals_dir)

# LDA için görselleştirme (tek boyut)
plt.figure(figsize=(10, 6))
plt.scatter(X_train_lda[:, 0], np.zeros_like(X_train_lda[:, 0]), c=y_train, cmap="viridis", edgecolor="k", alpha=0.7)
plt.colorbar(label="Sınıf")
plt.title("LDA - 1 Bileşen")
plt.xlabel("Bileşen 1")
plt.yticks([])
plt.savefig(os.path.join(lda_visuals_dir, "LDA_Scatter.png"))
plt.close()

# No Reduction için görselleştirme
plot_scatter(X_train, y_train, "No Reduction - Orijinal Veri", "NoReduction_Scatter.png", noreduction_visuals_dir)

# ===================== Model Tanımları ve Değerlendirme =====================
models = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
}

def evaluate_models(X_train, X_test, y_train, y_test, method_name):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt=f"Evaluation Results - {method_name}", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion Matrix Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {method_name} - {model_name}")
        cm_filename = f"{method_name}_{model_name}_ConfusionMatrix.png"
        if method_name == "PCA":
            plt.savefig(os.path.join(pca_visuals_dir, cm_filename))
        elif method_name == "LDA":
            plt.savefig(os.path.join(lda_visuals_dir, cm_filename))
        elif method_name == "t-SNE":
            plt.savefig(os.path.join(tsne_visuals_dir, cm_filename))
        elif method_name == "No Reduction":
            plt.savefig(os.path.join(noreduction_visuals_dir, cm_filename))
        else:
            plt.savefig(os.path.join(visuals_dir, cm_filename))
        plt.close()
        
        # Add results to PDF
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(200, 10, txt=f"Model: {model_name}", ln=True, align='L')
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Accuracy: {acc:.4f}", ln=True, align='L')
        pdf.ln(5)
        pdf.cell(200, 10, txt="Classification Report:", ln=True, align='L')
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                metrics_str = ', '.join([f"{key}: {value:.2f}" for key, value in metrics.items()])
                pdf.cell(200, 10, txt=f"  {label}: {metrics_str}", ln=True, align='L')
        pdf.ln(10)
        
        # Add confusion matrix image to PDF
        if method_name == "PCA":
            pdf.image(os.path.join(pca_visuals_dir, cm_filename), w=100)
        elif method_name == "LDA":
            pdf.image(os.path.join(lda_visuals_dir, cm_filename), w=100)
        elif method_name == "t-SNE":
            pdf.image(os.path.join(tsne_visuals_dir, cm_filename), w=100)
        elif method_name == "No Reduction":
            pdf.image(os.path.join(noreduction_visuals_dir, cm_filename), w=100)
        else:
            pdf.image(os.path.join(visuals_dir, cm_filename), w=100)
        pdf.ln(10)
        
        results[model_name] = acc
    
    # Save PDF report
    pdf.output(os.path.join(report_dir, f"{method_name}_Evaluation_Report.pdf"))
    return results

# Değerlendirme
print("==== Boyut İndirgeme Olmadan ====")
evaluate_models(X_train, X_test, y_train, y_test, "No Reduction")

print("\n==== PCA ====")
evaluate_models(X_train_pca, X_test_pca, y_train, y_test, "PCA")

print("\n==== LDA ====")
evaluate_models(X_train_lda, X_test_lda, y_train, y_test, "LDA")

print("\n==== t-SNE ====")
evaluate_models(X_train_tsne, X_test_tsne, y_train, y_test, "t-SNE")

# ===================== Kıyaslama Görselleştirmesi =====================
methods = ["No Reduction", "PCA", "LDA", "t-SNE"]
accuracies = {}

# Model başarımlarını toplama
accuracies["No Reduction"] = evaluate_models(X_train, X_test, y_train, y_test, "No Reduction")
accuracies["PCA"] = evaluate_models(X_train_pca, X_test_pca, y_train, y_test, "PCA")
accuracies["LDA"] = evaluate_models(X_train_lda, X_test_lda, y_train, y_test, "LDA")
accuracies["t-SNE"] = evaluate_models(X_train_tsne, X_test_tsne, y_train, y_test, "t-SNE")

# Başarımları görselleştirme
for model_name in models.keys():
    plt.figure(figsize=(10, 6))
    for method in methods:
        plt.bar(method, accuracies[method][model_name], label=method)
    plt.title(f"Model Accuracy Comparison - {model_name}")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(os.path.join(comparison_visuals_dir, f"Comparison_{model_name}.png"))
    plt.close()

print("Tüm işlemler tamamlandı. Çıktılar 'Output/Visuals' ve raporlar 'Output/Reports' klasörüne kaydedildi.")
