# Sonuçların Değerlendirilmesi (MAE, RMSE, Grafik)
import numpy as np 
import matplotlib.pyplot as plt
from keras.models import load_model  # Modern Keras kullanımı
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# 1. Verileri ve Modeli Yükleyelim
X_test = np.load("X_test.npy")  # .npy uzantısı düzeltildi
y_test = np.load("y_test.npy")  # dosya adı küçük y_test.npy yapıldı

# Proje dizinindeki dosya adlarına uygun yüklüyoruz
model = load_model("model.h5")   # veya 'lstm_model.h5'
scaler = joblib.load("scaler.save")  # ADIM2'de kaydettiğimiz scaler adı

# 2. Tahmin Yapalım
y_predict_scaled = model.predict(X_test)

# 3. Geri Ölçeklendirme (Inverse Transform)
y_pred = scaler.inverse_transform(y_predict_scaled)

# y_test tek boyutluysa 2D formata getirip geri ölçeklendiriyoruz
if y_test.ndim == 1:
    y_test = y_test.reshape(-1, 1)
y_true = scaler.inverse_transform(y_test)

# 4. Hata Metrikleri
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# f-string formatı eklendi
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# 5. İlk 200 Tahminin Grafiğini Çizdirme ve Kaydetme
plt.figure(figsize=(12, 6))
plt.plot(y_true[:200], label="Gerçek", linewidth=2)
plt.plot(y_pred[:200], label="Tahmin", linestyle="--")
plt.title("Gerçek vs Tahmin (İlk 200 Saat)")
plt.xlabel("Saat")
plt.ylabel("Güç Tüketimi (kW)")
plt.legend()

# Codespaces ortamında resmi sol menüye kaydetmek için
plt.savefig("tahmin_vs_gercek.png", dpi=300, bbox_inches="tight")
print("Grafik 'tahmin_vs_gercek.png' olarak kaydedildi!")

# Görsel pencere için
plt.show()