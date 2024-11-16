
# Boyut İndirgeme Tekniklerinin Sınıflandırmaya Etkisi

Bu proje, UCI Machine Learning Repository'den alınan **Breast Cancer Wisconsin (Diagnostic)** veri kümesini kullanarak, farklı boyut indirgeme yöntemlerinin sınıflandırma performansına etkilerini analiz eder. Projede PCA, LDA ve t-SNE gibi boyut indirgeme teknikleri ile Logistic Regression, Random Forest, SVM, KNN ve Naive Bayes algoritmaları birlikte değerlendirilmiştir.

## Proje Hedefi

Bu proje kapsamında:
1. Yüksek boyutlu verinin boyut indirgeme teknikleriyle düşük boyutlara indirgenmesi ve sınıflandırma performansının analiz edilmesi.
2. PCA, LDA ve t-SNE yöntemlerinin avantaj ve dezavantajlarının görselleştirme ve performans metrikleriyle ortaya konması.
3. Logistic Regression, Random Forest, SVM, KNN ve Naive Bayes gibi algoritmaların boyut indirgeme ile performans değişimlerinin karşılaştırılması.

---

## Kullanılan Teknolojiler ve Kütüphaneler

Bu proje Python programlama dili ile geliştirilmiştir ve aşağıdaki kütüphaneleri kullanmaktadır:

- **Pandas**: Veri manipülasyonu ve analizi.
- **Numpy**: Sayısal hesaplamalar.
- **Matplotlib** ve **Seaborn**: Veri görselleştirme.
- **Scikit-learn**: Makine öğrenmesi algoritmaları, boyut indirgeme teknikleri ve değerlendirme metrikleri.
- **Joblib**: Model kaydetme ve yükleme.

---

## Proje Yapısı

```
📦 Proje Klasörü
├── Data/
│   └── wdbc.data            # Veri kümesinin yer alacağı klasör
├── Output/
│   ├── Models/              # Eğitilmiş modellerin saklandığı klasör
│   ├── Visuals/             # Görsellerin saklandığı klasör
│   └── results_summary.txt  # Sınıflandırma sonuçlarının metin dosyası
├── main.py                  # Projenin ana Python dosyası
├── requirements.txt         # Python kütüphanelerini içeren dosya
└── README.md                # Proje açıklamaları
```

---

## Veri Kümesi Hakkında

- **Kaynak**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Adı**: Breast Cancer Wisconsin (Diagnostic)
- **Gözlem Sayısı**: 569
- **Özellik Sayısı**: 30 (sayısal)
- **Hedef Değişken**: `Diagnosis` (Benign [0], Malignant [1])

---

## Çalıştırma

### 1. Gerekli Ortamı Hazırlayın
Python sürümünüzün 3.8 veya üzeri olduğundan emin olun. Gerekli kütüphaneleri yüklemek için aşağıdaki komutu kullanabilirsiniz:
```bash
pip install -r requirements.txt
```

### 2. Veri Kümesini Ekleyin
`Data` klasörüne `wdbc.data` dosyasını ekleyin. Veri kümesini [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) üzerinden indirebilirsiniz.

### 3. Ana Dosyayı Çalıştırın
Projenin ana dosyasını çalıştırmak için terminalde aşağıdaki komutu kullanın:
```bash
python main.py
```

### 4. Çıktılara Erişim
- **Sonuçlar**: `Output/results_summary.txt` dosyasında sınıflandırma başarımları bulunur.
- **Görseller**: PCA, LDA ve t-SNE ile oluşturulan dağılım grafiklerini `Output/Visuals` klasöründe bulabilirsiniz.
- **Modeller**: Eğitilmiş modeller `Output/Models` klasöründe saklanır.

---

## Proje Çıktıları

### 1. Sınıflandırma Başarımları
Boyut indirgeme olmadan ve her boyut indirgeme yöntemiyle yapılan sınıflandırma işlemlerinin başarımları karşılaştırılmıştır. Sonuçlar aşağıdaki dosyada bulunabilir:
```
Output/results_summary.txt
```

### 2. Görselleştirmeler
PCA, LDA ve t-SNE yöntemleriyle indirgenmiş verilerin dağılım grafiklerini `Output/Visuals` klasöründe bulabilirsiniz. Örnek görseller:
- `PCA_Scatter.png`
- `LDA_Scatter.png`
- `tSNE_Scatter.png`

### 3. Nihai Model
Random Forest modeli, verinin tamamı üzerinde eğitilmiş ve `Output/Models` klasöründe kaydedilmiştir.

---

## Katkılar

Herhangi bir katkıda bulunmak veya hata bildirmek isterseniz lütfen bir **pull request** gönderin veya bir **issue** açın. Bu projeyi geliştirmek için önerilere açığız.

---

## Lisans

Bu proje **MIT Lisansı** altında yayımlanmıştır. Daha fazla bilgi için `LICENSE` dosyasına göz atabilirsiniz.
