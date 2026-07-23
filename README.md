# ltsm-ile-enerji-t-ketimi-tahmini-btk-akademi-kurs-uygulama
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
pip install pandas veri bilimi numpy  umeric matplotlib görselleştirme scikit-learn makine öğrenmesi tensorflow derin öğrenme algoritması lstm için
"""
