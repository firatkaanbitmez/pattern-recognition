# Gerekli kütüphanelerin yüklenmesi
import pandas as pd

# Red ve White wine veri setlerinin yolları
red_wine_path = r'C:\Users\FIRAT\Desktop\pattern-recognition\wine-quality-analysis\data\winequality-red.csv'
white_wine_path = r'C:\Users\FIRAT\Desktop\pattern-recognition\wine-quality-analysis\data\winequality-white.csv'

# Veri setlerinin okunması
red_wine = pd.read_csv(red_wine_path, sep=';')
white_wine = pd.read_csv(white_wine_path, sep=';')

# Eksik veri kontrolü
print("\nRed Wine Veri Seti - Eksik Değer Kontrolü:")
if red_wine.isnull().sum().sum() == 0:
    print("Red Wine veri setinde eksik veri bulunmamaktadır.")
else:
    print("Red Wine veri setinde eksik veriler bulunmaktadır.")
    eksik_veriler_red = red_wine.isnull().sum()
    print(eksik_veriler_red[eksik_veriler_red > 0])  # Eksik verilerin detaylarını göster

print("\nWhite Wine Veri Seti - Eksik Değer Kontrolü:")
if white_wine.isnull().sum().sum() == 0:
    print("White Wine veri setinde eksik veri bulunmamaktadır.")
else:
    print("White Wine veri setinde eksik veriler bulunmaktadır.")
    eksik_veriler_white = white_wine.isnull().sum()
    print(eksik_veriler_white[eksik_veriler_white > 0])  # Eksik verilerin detaylarını göster
