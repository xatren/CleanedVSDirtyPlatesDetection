# Kaggle Competition: Cleaned vs Dirty Plates Classification

Bu repoda, temiz ve kirli tabak görüntülerini sınıflandırmak için oluşturduğumuz modelin kodu ve açıklamaları bulunmaktadır. Yarışma sırasında karşılaştığımız kısıtlamalar ve küçük veri seti nedeniyle özel bir yaklaşım benimsedik.

## İçindekiler

- [Giriş](#giriş)
- [Veri Seti](#veri-seti)
- [Yaklaşım](#yaklaşım)
- [Model Mimarisi](#model-mimarisi)
- [Veri Ön İşleme ve Artırma](#veri-ön-işleme-ve-artırma)
- [Eğitim](#eğitim)
- [Değerlendirme](#değerlendirme)
- [Sonuçlar](#sonuçlar)
- [Gereksinimler](#gereksinimler)
- [Teşekkür](#teşekkür)

## Giriş

Bu Kaggle yarışmasında, amaç temiz ve kirli tabak görüntülerini doğru bir şekilde sınıflandırabilen bir makine öğrenimi modeli geliştirmektir. Küçük veri seti ve yarışma ortamındaki kısıtlamalar (örneğin, internet erişiminin olmaması) nedeniyle, veri ve model seçimi konusunda özel zorluklarla karşılaştık.

## Veri Seti

Veri seti aşağıdaki gibidir:

- **Eğitim Seti**: 40 görüntü (20 temiz, 20 kirli)
- **Test Seti**: Etiketlenmemiş görüntüler (744)

Veri setinin küçük olması nedeniyle, eğitim verilerinin etkin boyutunu artırmak için yoğun veri artırma teknikleri uyguladık.

## Yaklaşım

Başlangıçta VGG16 gibi önceden eğitilmiş modellerle transfer öğrenme kullanmayı denedik. Ancak, yarışma ortamında internet erişimi olmadığından önceden eğitilmiş ağırlıkları indiremedik. Bu nedenle, sıfırdan bir Konvolüsyonel Sinir Ağı (CNN) tasarladık.

Ana adımlarımız:

1. **Veri Ön İşleme ve Artırma**: Yoğun veri artırma ile eğitim verilerini genişletme.
2. **Model Tasarımı**: Küçük veri setleri için uygun bir CNN mimarisi.
3. **Eğitim Stratejisi**: Aşırı uyumayı önlemek için düzenlileştirme tekniklerinin kullanımı.

## Model Mimarisi

CNN modelimiz aşağıdaki bileşenlerden oluşur:

- **Konvolüsyonel Katmanlar**: Artan filtre boyutlarına sahip üç konvolüsyon bloğu (32, 64, 128).
- **Batch Normalizasyonu**: Her konvolüsyon katmanından sonra uygulanır.
- **Havuzlama Katmanları**: Uzaysal boyutları azaltmak için MaxPooling katmanları.
- **Tam Bağlantılı Katmanlar**: 256 birimli ve ReLU aktivasyonlu bir yoğun katman.
- **Dropout Katmanı**: Aşırı uyumayı önlemek için %50 dropout.
- **Çıkış Katmanı**: Sigmoid aktivasyonlu tek nöronlu bir katman (ikili sınıflandırma için).

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

## Veri Ön İşleme ve Artırma

Veri setinin küçük olması nedeniyle, veri artırma tekniklerini agresif bir şekilde uygulayarak veri setinin boyutunu ve çeşitliliğini artırdık.

Uygulanan artırma teknikleri:

- **Döndürme**: 90 dereceye kadar
- **Genişlik ve Yükseklik Kaydırma**: %30'a kadar
- **Kesme Dönüşümleri**
- **Yakınlaştırma**
- **Yatay ve Dikey Çevirme**

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
```

## Eğitim

Modeli aşağıdaki strateji ile eğittik:

- **Optimizer**: 1e-4 öğrenme oranı ile Adam.
- **Kayıp Fonksiyonu**: Binary cross-entropy.
- **Metrikler**: Doğruluk (accuracy).
- **Callback'ler**:
  - **EarlyStopping**: Doğrulama kaybını izler, 15 epoch sabit kaldığında durur.
  - **ReduceLROnPlateau**: Doğrulama kaybı plato yaptığında öğrenme oranını azaltır.

Küçük veri seti nedeniyle `steps_per_epoch` parametresini belirtmedik ve Keras'ın otomatik olarak hesaplamasına izin verdik.

## Değerlendirme

Modeli aşağıdaki yöntemlerle değerlendirdik:

- **Confusion Matrix (Karmaşıklık Matrisi)**: Her sınıf üzerindeki performansı görselleştirmek için.
- **Classification Report (Sınıflandırma Raporu)**: Her sınıf için precision, recall, f1-score ve destek değerlerini sağlar.

```python
from sklearn.metrics import classification_report, confusion_matrix

Y_pred = model.predict(validation_generator)
y_pred = np.round(Y_pred).astype(int).flatten()
y_true = validation_generator.classes

cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix')
print(cm)

cr = classification_report(y_true, y_pred, target_names=validation_generator.class_indices.keys())
print('Classification Report')
print(cr)
```

## Sonuçlar

Karşılaştığımız zorluklara rağmen, model kısıtlamalar göz önüne alındığında tatmin edici sonuçlar elde etti. Agresif veri artırma ve düzenlileştirme teknikleri, aşırı uyumayı hafifletmeye yardımcı oldu.

## Gereksinimler

- Python 3.6 veya üstü
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- scikit-learn


## Teşekkür

- Bu yarışmayı ve veri setini sağladığı için Kaggle'a teşekkür ederiz.
