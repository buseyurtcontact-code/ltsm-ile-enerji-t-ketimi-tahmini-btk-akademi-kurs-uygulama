#gerçek haytta nasıl uugyualanacak gelecek tahmini kısmı 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib 
import pandas as pd

#model ve scaler yüklenmesi 
model = load_model("model.h5")
scaler = joblib.load("scaler.save")

#zaman serisinin yüklenmeesi 
df_hourly = pd.read_csv("df_hourly.csv", index_col=0, parse_dates=True)

#son 48 saati alalım 24 saatlik geçmişi kullanarak 24 saatlik geleceği tahmin edelim
last_48 = df_hourly.iloc[-48:].copy() #son 48 saatlik veriyi seç
last_24 = last_48.iloc[:24].values #son 48in ilk 24ü, modelin girdisi olacak 
real_next_24 = last_48.iloc[24:].values #son 48in son 24ü, model tahminleriylekarşılaştırılacak gerçek değerler 
#yani elimizdeki son 48 saatlik verinin ilk 24ü test edeceğiz ve son 24ün gerçek değerleriyle karşılaştıracağız

#normalize edilmiş veriyi alalım 
X_test = np.load("X_test.npy")
forecast_input = X_test[-1].copy() #test setinin son penceresi ileriye dönük tahminlerde başlangıç olacak 

# modelimiz ile 24 saatlik tahmin yap
future_predictions = []  # tahmin edilen kwH degerleri burada tutulur

for _ in range(24):  # 24 saatlik döngü başlatılır
    input_3d = forecast_input.reshape(1, forecast_input.shape[0], 1)  # model 3 boyutlu giris bekler -> (örnek sayısı, zaman adımı, öznitelik sayısı)
    
    next_scaled = model.predict(input_3d, verbose = 0)[0]  # modelin çıktısı ölçeklenmiş yani 0-1 arasında olur
    
    next_value = scaler.inverse_transform(next_scaled.reshape(1, -1))[0][0]  # orijinal değere dönüştürülür
    
    future_predictions.append(next_value)  # tahmin edilen gerçek ölçekteki değer listeye eklenir
    
    # yeni girdi penceresini güncelle, ilk degeri at, tahmin edilen değeri sona ekle
    forecast_input = np.vstack((forecast_input[1:], next_scaled.reshape(1, 1)))  # zaman penceresi bir adım kaydırılır

# karşılaştırmalı grafik
plt.figure()  # yeni bir grafik penceresi/figürü oluşturulur

plt.plot(real_next_24.flatten(), label = "Gerçek (Gelecek 24 saat)", linewidth = 2)  # gerçek gelecek 24 saatlik veriler çizdirilir

plt.plot(future_predictions, label = "Tahmin (Gelecek 24 saat)", linewidth = 2, linestyle = "--")  # tahmin edilen 24 saatlik değerler kesikli çizgiyle çizdirilir

plt.title("Gelecek 24 saat: Gerçek vs Tahmin")  # grafiğin başlığı ayarlanır

plt.xlabel("Saat")  # X ekseninin etiketi ayarlanır

plt.ylabel("kWh")  # Y ekseninin etiketi ayarlanır

plt.legend()  # grafik üzerine veri etiketleri (lejand) eklenir

plt.grid(True)  # grafiğe ızgara çizgileri eklenir

plt.savefig("gelecek_24saat_tahmini.png", dpi=300, bbox_inches="tight")  # görsel sanal ortama/klasöre kaydedilir

plt.show()  # grafik ekranda gösterilir