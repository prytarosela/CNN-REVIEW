
# **Implementasi Convolutional Neural Networks (CNN)**

## 📖 **Gambaran Umum**
Convolutional Neural Networks (CNN) adalah arsitektur deep learning yang sangat efektif dalam pengolahan data grid seperti gambar. CNN digunakan secara luas dalam berbagai aplikasi seperti klasifikasi gambar, deteksi objek, segmentasi gambar, dan lainnya.

Repository ini berisi:
1. **Review CNN**: Penjelasan teori CNN secara umum.
2. **Point 2**: Kode implementasi klasifikasi biner gambar menggunakan dataset custom.
3. **Point 3**: Kode implementasi klasifikasi multi-kelas menggunakan dataset CIFAR-10.

---

## 🔍 **Penjelasan CNN**

### Apa Itu CNN?
Convolutional Neural Networks (CNN) adalah jenis jaringan saraf tiruan yang dirancang untuk memproses data dengan struktur grid, seperti gambar. CNN terdiri dari beberapa jenis lapisan utama:
- **Convolutional Layer**: Mengekstraksi fitur dari gambar, seperti tepi, pola, atau tekstur.
- **Pooling Layer**: Mengurangi dimensi spasial data untuk menghemat komputasi.
- **Fully Connected Layer**: Menggunakan fitur yang diekstraksi untuk melakukan klasifikasi.

### Alur Dasar CNN:
1. Input gambar dengan dimensi tertentu.
2. Convolution dan pooling untuk mengekstraksi fitur.
3. Flattening data menjadi vektor 1D.
4. Fully connected layers untuk klasifikasi akhir.

---

## 🔎 **Point 2: Implementasi Klasifikasi Biner Gambar**

### **Deskripsi**
Implementasi CNN untuk klasifikasi biner gambar menggunakan dataset custom. Model bertujuan untuk memprediksi apakah sebuah gambar termasuk dalam salah satu dari dua kelas tertentu.

### **Struktur Model**
```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
```

- **Convolutional Layer**: Ekstraksi fitur dengan 32 filter ukuran 3x3.
- **Pooling Layer**: MaxPooling ukuran 2x2 untuk mengurangi dimensi data.
- **Activation**: ReLU untuk convolutional layer dan sigmoid untuk klasifikasi biner.

### **Augmentasi Data**
```python
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
```
Augmentasi data digunakan untuk memperbesar variasi dataset pelatihan:
- **Rescale**: Menormalisasi piksel gambar ke rentang [0, 1].
- **Shear, Zoom, Flip**: Membuat dataset lebih beragam.

### **Training Model**
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(training_set, validation_data=validation_set, epochs=25)
```
- Optimizer: Adam.
- Loss function: Binary Crossentropy.
- Jumlah epoch: 25.

### **Hasil Training**
- Akurasi Pelatihan: XX%.
- Akurasi Validasi: XX%.

---

## 🔎 **Point 3: Implementasi Klasifikasi Multi-Kelas (CIFAR-10)**

### **Deskripsi**
Implementasi CNN untuk klasifikasi multi-kelas menggunakan dataset CIFAR-10. Dataset ini berisi gambar berwarna 32x32 dengan 10 kategori, seperti pesawat, mobil, burung, kucing, dll.

### **Struktur Model**
```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
```
- Tiga lapisan convolutional dan pooling untuk ekstraksi fitur.
- Dropout untuk mencegah overfitting.
- Lapisan terakhir memiliki 10 neuron dengan softmax untuk klasifikasi multi-kelas.

### **Training Model**
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(training_set, validation_data=validation_set, epochs=50)
```
- Optimizer: Adam.
- Loss function: Categorical Crossentropy.
- Jumlah epoch: 50.

### **Evaluasi**
- Evaluasi model pada data uji:
  ```python
  test_loss, test_accuracy = model.evaluate(test_set)
  ```
- Prediksi gambar baru:
  ```python
  prediction = model.predict(image)
  ```

### **Hasil Training**
- Akurasi Pelatihan: 75.29%.
- Akurasi Validasi: 70.45%.
- Akurasi Data Uji: 70.46%.

---

## 🧪 **Hasil dan Observasi**
1. **Point 2 (Klasifikasi Biner):**
   - Akurasi validasi mencapai XX%.
   - Augmentasi data membantu mengurangi overfitting.
   - Model dapat ditingkatkan dengan dataset lebih besar.

2. **Point 3 (Klasifikasi Multi-Kelas):**
   - Akurasi uji sebesar 70.46% pada dataset CIFAR-10.
   - Dropout layer mungkin efektif dalam mencegah overfitting, meskipun hasilnya menunjukkan sedikit gap antara akurasi pelatihan dan uji.
   - Model cukup baik untuk mengenali kategori berbeda pada CIFAR-10.

---

## 🚀 **Cara Menjalankan Kode**
1. Clone repository ini:
   ```bash
   git clone https://github.com/prytarosela/CNN-REVIEW.git
   ```
2. Masuk ke folder repository:
   ```bash
   cd CNN-REVIEW
   ```
3. Jalankan file `.ipynb` di Jupyter Notebook atau Google Colab.
4. Pastikan semua dependensi telah terinstall, seperti TensorFlow dan Keras:
   ```bash
   pip install tensorflow keras
   ```

---

## 📚 **Referensi**
- [Deep Learning: Convolutional Neural Networks](https://www.megabagus.id/deep-learning-convolutional-neural-networks/)
- [Deep Learning: CNN dan Pikselasi](https://www.megabagus.id/deep-learning-convolutional-neural-networks-pixilasi/)
- [Materi Praktikum AI - CNN](https://modul-praktikum-ai.vercel.app/Materi/4-convolutional-neural-network)
