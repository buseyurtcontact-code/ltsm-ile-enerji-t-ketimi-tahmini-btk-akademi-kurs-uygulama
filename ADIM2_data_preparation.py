# verileri hazırlama kısmı 
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # Verileri ölçeklemek yani normalize etmek
import pandas as pd
import joblib  # Model ve işlem nesnelerini kaydetmek ve yüklemek

# Daha önce hazırlanan saatlik veri dosyasını oku
# index_col=0 veya index_col="datetime" diyerek ilk sütunu indeks yapıyoruz
# parse_dates=["datetime"] ile tarih-saat verisini doğrudan datetime formatına çeviriyoruz
df_hourly = pd.read_csv(
    "df_hourly.csv",
    index_col="datetime",
    parse_dates=["datetime"]
)

# NaN değerleri temizle
df_hourly.dropna(inplace=True)

# kabaca pandas ile veri yükleme yapılır eğer ml ysa ile ilgili çalışıyorsak numpy'a doğru çeviririz 
# yani veri bilimi işlemleri bitmeye doğru numpya geç

# pandas dataframe formatından numpy array formatına çevrilir
values = df_hourly.values.reshape(-1, 1)

# normalizasyon (0-1 arasına sıkıştırma)
scaler = MinMaxScaler() # 0-1 arasına ölçeklemek için gerekli sınıf
scaled = scaler.fit_transform(values) # Önce veriye göre min-max değerlerini hesaplıyor sonrasında dönüştürüyor.
# neden normalizasyon: lstm gibi modellerde, modelin daha hızlı ve stabil öğrenmesi için önemli
# 1. Aktivasyon Fonksiyonları: Sigmoid/Tanh kapılarının doygunluğa ulaşıp tıkanmasını önler.
# 2. Hızlı ve Stabil Öğrenme: Ağırlık güncellemelerindeki (gradient descent) savrulmaları engeller.
# 3. Eşit Ölçekleme: Farklı büyüklükteki değişkenlerin birbirini ezmesini önler, ortak skalaya getirir.

# scaler kaydetme
joblib.dump(scaler, "scaler.save") # test veya gerçek zamanlı tahminde aynı ölçekleyiciyi kullanmak için kaydettik

# ---------------------------------------------------------
# SLIDING WINDOW (KAYAN PENCERE) FONKSİYONU
# ---------------------------------------------------------

def create_sliding_window(data, window_size = 24):
    """
    data: Normalleştirilmiş zaman serisi verisi (1D veya 2D NumPy dizisi)
    window_size: Girdi olarak kullanılacak geçmiş adım sayısı (örn: son 24 saat)
    """
    X, y = [], []
    
    # Döngü, pencerenin dizinin sonunu aşmaması için (toplam_uzunluk - pencere_boyutu) kadar döner
    for i in range(len(data) - window_size):
        # i'den başlayıp (i + window_size)'a kadar olan geçmiş verileri X'e (girdi) ekle
        X.append(data[i : i + window_size])
        
        # Tam pencerenin bittiği andaki hedef değeri (25. saatteki değer) y'ye (hedef) ekle
        y.append(data[i + window_size])
        
    # Python listelerini LSTM'in işleyebileceği NumPy dizilerine çevirip döndürüyoruz
    return np.array(X), np.array(y)

# Mantık: LSTM modelimize girdi (X) = geçmiş 24 saat, çıktı (y) = 25. saatteki değer

"""
Örnek Mantık (window_size = 3 için):
Veri seti: [1, 2, 3, 4, 5, 6, 7, 8, 9]

1. Adım: X = [1, 2, 3] -> y = 4
2. Adım: X = [2, 3, 4] -> y = 5
...şeklinde pencere kayarak devam eder.
"""

# giriş ve çıkış verilerini oluştur
window_size = 24  # Son 24 saatin verisine bakarak bir sonraki saati tahmin edeceğiz
X, y = create_sliding_window(scaled, window_size)

# train test split - eğitim ve test ayrımı
# ÖNEMLİ: Zaman serilerinde veri rastgele KARIŞTIRILMAZ (shuffle=False). 
# Zamansal sırayı bozmamak için ilk %80 ile model eğitilir, kalan %20 ile test edilir.
split = int(len(X) * 0.8)

# Veriyi indeks bazlı dilimleme (slicing) ile %80 eğitim ve %20 test olarak ayırıyoruz
X_train, X_test = X[:split], X[split:]  # İlk %80 eğitim girdisi, kalan %20 test girdisi
y_train, y_test = y[:split], y[split:]  # İlk %80 eğitim hedefi, kalan %20 test hedefi

# şekil (shape) kontrolü
# LSTM modeli 3 boyutlu veri bekler: (Örnek Sayısı, Zaman Adımı/Window Size, Özellik Sayısı)
print(f"X_train shape: {X_train.shape}") # (27315, 24, 1) -> (örnek sayısı, zaman adımı, özellik sayısı)
print(f"y_train shape: {y_train.shape}") # (27315, 1) bunlar bouytlar y çıktı bpyutu x girdilerin boyutu


# kaydet
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
