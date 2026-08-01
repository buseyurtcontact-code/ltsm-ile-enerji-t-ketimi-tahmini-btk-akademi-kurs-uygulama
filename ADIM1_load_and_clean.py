# Veri yükleme ve temizleme
import pandas as pd # Veri işleme ve analiz
import numpy as np # Sayısal işlemler
import matplotlib.pyplot as plt # Grafik çizimi

# Dosya noktalı virgül (;) ile ayrıldığı için sep=';' parametresini ekliyoruz
# Eksik veriler '?' karakteri ile temsil edildiği için na_values='?' yapıyoruz
df = pd.read_csv('household_power_consumption.txt',
                 sep=';',
                 low_memory=False,
                 na_values="?")

# Date ve Time sütunlarını birleştirip güncel standart olan pd.to_datetime ile dönüştürüyoruz
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')

# Eski ve ayrı duran Date ve Time sütunlarını siliyoruz
df.drop(columns=['Date', 'Time'], inplace=True)

# Datetime sütununu tablonun indeksi yaptık 
df.set_index('datetime', inplace=True)

# İlk 5 satırı görüntüleyinpython ADIM1_load_and_clean.py
print(df.head())

# ---------------------------------------------------------
# GLOBAL ACTIVE POWER TEMİZLEME VE DÖNÜŞTÜRME
# ---------------------------------------------------------

# Global_active_power sütununu sayısal (float) veri tipine dönüştürüyoruz
# errors='coerce': Sayıya çevrilemeyen veya gözden kaçan hatalı karakterleri otomatik NaN yapar
df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"], 
    errors="coerce"
)

# Sadece hedef sütunumuz olan Global_active_power içindeki NaN (eksik) değerleri temizliyoruz
# subset parametresi sayesinde tablonun kalanını bozmadan sadece bu sütunu baz alarak eksik satırları siler
df.dropna(subset=["Global_active_power"], inplace=True)

# Son halini kontrol etmek için ilk 5 satırı ekrana bastıralım
print("\nTemizlenmiş Veri Seti (İk 5 Satır):")
print(df.head())

# Verinin tipini ve eksik değer kalıp kalmadığını doğrulamak için özet bilgi
print("\nVeri Tipi ve Özeti:")
print(df["Global_active_power"].info())

# Saatlik ortalamaya göre yeniden örnekleme (Resampling)
# '1h' veya 'h' güncel Pandas standartıdır
df_hourly = df[['Global_active_power']].resample('h').mean()

# Resample sonrası oluşabilecek olası boş saatleri (forward) bir önceki değerle dolduruyoruz
df_hourly = df_hourly.ffill()

# İlk 5 satırı kontrol edelim
print("\nSaatlik Ortalamalara Göre Yeniden Örneklenmiş Veri:")
print(df_hourly.head())

# Zaman serisini görselleştir
plt.figure(figsize=(12, 6)) # Grafiğin boyutunu daha okunabilir yapmak için genişlettik
plt.plot(df_hourly, label="Saatlik Enerji Tüketimi")
plt.title("Enerji Tüketimi")
plt.xlabel("Zaman")
plt.ylabel("kW")
plt.legend()
plt.show()

# plt.show() masaüstünde grafik otomatik açar biz sanal ortamda çalıştığımız için aşağıdaki gibi yazmalıyız 
plt.savefig("enerji_tuketimi.png", dpi=300, bbox_inches='tight')
print("Grafik 'enerji_tuketimi.png' olarak kaydedildi!")

# Saatlik yeniden örneklenmiş veriyi kaydet
# index=True diyerek datetime indeksimizi de CSV'ye yazmış oluyoruz (Varsayılanı zaten True'dur)
df_hourly.to_csv("df_hourly.csv", index=True)