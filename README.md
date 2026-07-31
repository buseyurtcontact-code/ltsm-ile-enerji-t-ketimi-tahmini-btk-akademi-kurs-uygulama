# lstm-ile-enerji-t-ketimi-tahmini-btk-akademi-kurs-uygulama
"""
LSTM ile Enerji Tüketimi Tahmini

Problem tanımı:
- Enerji tüketimi tahmini, elektrik üretim planlama, şebeke dengeleme, tüketim öngörülmesi ve faturalama optimizasyonu
- Geçmiş tüketim verilerine bakarak gelecek tüketim verilerini tahmin etmek
- Amaç: geçmiş enerji tüketim verisinden ileriye dönük enerji tüketim tahmini ve enerji planlaması

Data:
- https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- global_active_power (kilowatt cinsinden toplam aktif güç)

Kullanılan Araçlar ve Teknolojiler:
- LSTM: TensorFlow/Keras
  - RNN türüdür,
  - Gradyan sönmesi problemine çözüm olarak geliştirildi
  - unutma (forget) kapısı, girdi (input) kapısı, çıktı (output) kapısı

Plan/Program:
    - veri yükleme ve temizleme csv dosyasını ; ile okuyacağız ve date-time gibi columnları birleştireceğiz

    - yeniden örnekleme (resampling) dakikalık verileri saatlik ortalama değerlere dönüştüreceğiz 

    - kayan pencere (sliding window) oluşturma lstm girişleri için son 24 saate karşılık bir sonraki saatin değeri olacak şekilde x ve y değerlerini oluşturacağız*

    - veri ölçekleme (scaling) 0 ile 1 arasına normalizasyon işlemi yapacağız 

    - LSTM modelimizin eğitimi lstm modeli oluşturacağız 

    - performans analizi gerçekleştirme test verisi üzerinden tahminler yapıp gerçek değerler ile karşılaştırma yapacağız ve MSE  rootmse mean absolute error vb farklı değerlendirme metriklerini kullanarak değerlendirme yapacağız 

    - gelecek tahmini önümüzdeki 24 saatlik geleceği tahmin etmeye çalışacağız 

    install libraries: freeze
pip install
pandas veri bilimi 
numpy  numeric
matplotlib görselleştirme
scikit-learn makine öğrenmesi 
tensorflow derin öğrenme algoritması lstm için

## Model Eğitim Kayıp Grafiği (Loss Graph)

![Model Kayıp Grafiği](loss_grafigi.png)

### Grafik Değerlendirmesi:
* **Train vs Val Loss:** Hem Eğitim Kaybı (Train Loss) hem de Doğrulama Kaybı (Validation Loss) ilk epoch'lardan itibaren kararlı bir şekilde düşmüştür.
* **Overfitting Kontrolü:** İki çizginin birbirine yakın seyretmesi modelin ezber yapmadığını (overfitting olmadığını) ve veriyi genelleştirebildiğini gösterir.
* **Early Stopping:** Model 14. epoch civarında doğrulama kaybının (val_loss ≈ 0.006) en düşük seviyeye ulaşmasıyla eğitimi durdurmuş ve en iyi ağırlıkları kaydetmiştir.
* model kayıp grafiğide eğitim kaybı ve doğrulama kaybıı birbirine yakın ve düşük seviyede olması gerekir, eğer aralarındaki fark çoksa lstmdeki katman sayısı arttırılabilir 
### Model Kayıp Grafiği Analizi
* **İdeal Durum:** Model kayıp grafiğinde Eğitim Kaybı (Train Loss) ve Doğrulama Kaybı (Validation Loss) değerlerinin birbirine yakın ve düşük seviyede olması beklenir.
* **Model Kapasitesi & Yetersiz Öğrenme (Underfitting):** Eğer her iki kayıp değeri de yüksek kalırsa, modelin öğrenme kapasitesini artırmak için LSTM katman sayısı veya nöron sayısı yükseltilebilir.
* **Ezberleme (Overfitting):** Eğitim kaybı düşerken doğrulama kaybı yüksek kalır ve iki çizgi arasındaki fark açılırsa model ezber yapıyor demektir; bu durumda Dropout katmanları veya regülasyon teknikleri uygulanmalıdır.
"""
