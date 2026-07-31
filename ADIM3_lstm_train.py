# önceki derste veri setini yükledik eksik veirler ölçekleme ve lstm için hazır hale getirmek için windoiwng yapmnıştık şimdi eğitim ve test aşamalarıa giriş yapıyoruz 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# veriyi yukle
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# modeli tanimla
model = Sequential() #bunun içine model katmanları sıralı bir şekilde bağlanır 

# LSTM Katmanı (64 hücre) # 
model.add(LSTM(
    64, #64 tane lstm hücresi ile başladık 
    activation="tanh", #tanh aktivasyon fonksiyonunu kullandık, varsayılan zaman serileri için yaygın olarak kullanılır 
    input_shape=(X_train.shape[1], X_train.shape[2]) #[24,1] -> zaman adımı, öznitelik sayısı
))

# dense layer ekle
model.add(Dense(1)) #tek çıkışlı tam bağlantılı bir katman, sadece 1 saatlik enerji tahimini yapar, genelde bir adım sonrasını tahmin etmek isteriz (2 adım y da 3 adım sonrası değil)

# model compile (derleme) 
model.compile(
    optimizer="adam",  
#yaygı olarak kullaılan optimizasyon algoritması adam foksiyonu kullandık, hızlıdır adaptif öğrenir (adaptif öğrenme: # Adaptif Öğrenme (Adaptive Learning): Optimizatörün (örn: Adam), eğitimin gidişatına ve 
# hataya göre öğrenme oranını (adımlarının büyüklüğünü) otomatik olarak ayarlamasıdır. 
# Başta büyük adımlarla hızlı öğrenir, hedefe yaklaştıkça hassas ayar için adımları küçültür.
# # Gradient Descent İlgisi: Klasik Gradient Descent sabit adımlarla (sabit learning rate) 
# hatayı düşürmeye çalışırken; Adam gibi adaptif yöntemler, gradyanların (türevlerin) 
# büyüklüğüne göre her parametre için adım boyutunu anlık olarak optimize eder.)
)

# erken durdurma callback
#
early_stop = EarlyStopping(
    monitor="val_loss",  #doğrulama kaybı izlenir-takip edilir 
    patience=5, #art arda 5 epoch boyunca doğrulama kaybı iyileşmezse eğitim durur 
    restore_best_weights=True #böylece eğitim sırasındaki en iyi ağırlıklar(minimum validation loss yani en düşük doğrulama kaybı yai gerçek değer ile tahmin edilen değer farkının minimumu gibi düşün) geri yüklenir 
)

# eğitimi başlat (Epochs 20 yapıp erken durdurmaya bıraktık)
history = model.fit(
    X_train, y_train, #eğitim verileri
    validation_data=(X_test, y_test), #validation için test verilerini kullandık 
    epochs=20, #eğitimi 20 kez tekrarladık
    batch_size=32, #her bir eğitim adımında 32 örnek(sample) işlenir yani toplam veri setini 32lik örneklere bölüp işliyoruz, bu bir epoch eder 
    callbacks=[early_stop], #eğitim sırasında erken durdurma 
    verbose=1 #eğitim sırasında detaylı çıktılar yazılır 
)

# MODELLERİ KAYDETME (Çok Önemli Adım!)
model.save("model.h5")
print("Model başarıyla 'model.h5' olarak kaydedildi!")

# kayıp grafiği çizdirme
plt.plot(history.history["loss"], label="Eğitim Kaybı")
plt.plot(history.history["val_loss"], label="Doğrulama Kaybı")
plt.title("Model Kayıp Grafiği")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.savefig("loss_grafigi.png") # Grafiği de kaydeder
plt.show()

#kayı fonksiyonu grafiğinde
#Hem Eğitim Kaybı (Train Loss) hem de Doğrulama Kaybı (Validation Loss) ilk epoch'lardan itibaren çok güzel düşmüş ve 0.006 - 0.008 MSE civarında oldukça kararlı bir dengeye oturmuş.
#Grafik çizgilerinin birbirine yakın seyretmesi de modelinin ezber yapmadığını (overfitting olmadığını) ve gayet iyi öğrendiğini gösteriyor.
