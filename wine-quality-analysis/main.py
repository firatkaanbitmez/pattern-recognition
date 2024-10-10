import os
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Backend sorunu için matplotlib'in backend'ini değiştiriyoruz
import matplotlib
matplotlib.use('Agg')  # GUI olmadan grafiklerin oluşturulması için
current_dir = os.path.dirname(os.path.abspath(__file__))

# Output klasörünün oluşturulması ve eski dosyaların silinmesi
output_dir = os.path.join(current_dir, 'output')
if os.path.exists(output_dir):
    # Klasör içindeki dosyaları temizler
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
print(f"'output' klasörü '{output_dir}' konumunda başarıyla oluşturuldu.")

# Veri setlerinin yolları
red_wine_path = r'C:\Users\FIRAT\Desktop\pattern-recognition\a\data\winequality-red.csv'
white_wine_path = r'C:\Users\FIRAT\Desktop\pattern-recognition\a\data\winequality-white.csv'

# Veri setlerinin okunması
red_wine = pd.read_csv(red_wine_path, sep=';')
white_wine = pd.read_csv(white_wine_path, sep=';')

# 1) Veri kümesinin genel tanımlaması:
print("Veri seti, 2009 yılında Paulo Cortez ve ekibi tarafından oluşturulmuştur.")

# 2) Veri kümesi kaç sınıftan oluşmaktadır?
print("\nRed Wine Sınıf Dağılımı:")
print(red_wine['quality'].value_counts())
print("\nWhite Wine Sınıf Dağılımı:")
print(white_wine['quality'].value_counts())

# 3) Veri seti dengeli mi?
# Sınıf dağılımını görselleştirme
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(x='quality', data=red_wine, ax=ax[0])
ax[0].set_title('Red Wine Quality Distribution')
sns.countplot(x='quality', data=white_wine, ax=ax[1])
ax[1].set_title('White Wine Quality Distribution')
plt.savefig(os.path.join(output_dir, 'quality_distribution.png'))  # Grafiği kaydetme

# 4) Veri setinde kaç örnek yer almaktadır?
print(f"\nRed Wine Veri Setinde {red_wine.shape[0]} örnek ve {red_wine.shape[1]} özellik var.")
print(f"\nWhite Wine Veri Setinde {white_wine.shape[0]} örnek ve {white_wine.shape[1]} özellik var.")

# 5) Veri setinde eksik/boş değer var mı?
print("\nRed Wine Eksik Veri Kontrolü:")
print(red_wine.isnull().sum())
print("\nWhite Wine Eksik Veri Kontrolü:")
print(white_wine.isnull().sum())

# 6) Veri kümesindeki öznitelikler ve yapılarına genel bakış:
print("\nRed Wine Dataset - Info:")
print(red_wine.info())
print("\nWhite Wine Dataset - Info:")
print(white_wine.info())

# 7) Temel istatistiksel özellikler (Ortalama, Min, Max)
print("\nRed Wine İstatistiksel Özeti:")
print(red_wine.describe())
print("\nWhite Wine İstatistiksel Özeti:")
print(white_wine.describe())

# 8) Box plot grafikler ile özniteliklerin dağılımı
plt.figure(figsize=(10, 6))
sns.boxplot(data=red_wine)
plt.title('Red Wine Özniteliklerinin Box Plot Grafiği')
plt.xticks(rotation=90)
plt.savefig(os.path.join(output_dir, 'red_wine_boxplot.png'))  # Grafiği kaydetme

plt.figure(figsize=(10, 6))
sns.boxplot(data=white_wine)
plt.title('White Wine Özniteliklerinin Box Plot Grafiği')
plt.xticks(rotation=90)
plt.savefig(os.path.join(output_dir, 'white_wine_boxplot.png'))  # Grafiği kaydetme

# 9) Özniteliklerin histogram grafikleri
red_wine.hist(bins=15, figsize=(10, 8))
plt.suptitle('Red Wine Özniteliklerinin Histogramı')
plt.savefig(os.path.join(output_dir, 'red_wine_histogram.png'))  # Grafiği kaydetme

white_wine.hist(bins=15, figsize=(10, 8))
plt.suptitle('White Wine Özniteliklerinin Histogramı')
plt.savefig(os.path.join(output_dir, 'white_wine_histogram.png'))  # Grafiği kaydetme

# 10) Scatter plot ile veri kümesinin dağılımı
sns.pairplot(red_wine, diag_kind="kde")
plt.suptitle('Red Wine Scatter Plot Grafikleri')
plt.savefig(os.path.join(output_dir, 'red_wine_pairplot.png'))  # Grafiği kaydetme

sns.pairplot(white_wine, diag_kind="kde")
plt.suptitle('White Wine Scatter Plot Grafikleri')
plt.savefig(os.path.join(output_dir, 'white_wine_pairplot.png'))  # Grafiği kaydetme

# 11) Özniteliklerin sınıf ayrımı için scatter plotlar
sns.scatterplot(x='alcohol', y='quality', data=red_wine)
plt.title('Red Wine - Alcohol ve Quality İlişkisi')
plt.savefig(os.path.join(output_dir, 'red_wine_alcohol_quality_scatter.png'))  # Grafiği kaydetme

sns.scatterplot(x='alcohol', y='quality', data=white_wine)
plt.title('White Wine - Alcohol ve Quality İlişkisi')
plt.savefig(os.path.join(output_dir, 'white_wine_alcohol_quality_scatter.png'))  # Grafiği kaydetme

# 12) Violin plot ile sınıf yoğunluklarının görselleştirilmesi
plt.figure(figsize=(10, 6))
sns.violinplot(x='quality', y='alcohol', data=red_wine)
plt.title('Red Wine - Quality ve Alcohol Violin Plot')
plt.savefig(os.path.join(output_dir, 'red_wine_violinplot.png'))  # Grafiği kaydetme

plt.figure(figsize=(10, 6))
sns.violinplot(x='quality', y='alcohol', data=white_wine)
plt.title('White Wine - Quality ve Alcohol Violin Plot')
plt.savefig(os.path.join(output_dir, 'white_wine_violinplot.png'))  # Grafiği kaydetme

# 13) Sınıflandırma algoritmaları ile deneyler (KNN, SVM, Naive Bayes)
X_red = red_wine.drop('quality', axis=1)
y_red = red_wine['quality']
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_red, y_red, test_size=0.3, random_state=42)

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train_red, y_train_red)
y_pred_knn = knn.predict(X_test_red)
print(f"KNN Accuracy for Red Wine: {accuracy_score(y_test_red, y_pred_knn)}")

# SVM
svm = SVC()
svm.fit(X_train_red, y_train_red)
y_pred_svm = svm.predict(X_test_red)
print(f"SVM Accuracy for Red Wine: {accuracy_score(y_test_red, y_pred_svm)}")

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_red, y_train_red)
y_pred_nb = nb.predict(X_test_red)
print(f"Naive Bayes Accuracy for Red Wine: {accuracy_score(y_test_red, y_pred_nb)}")
