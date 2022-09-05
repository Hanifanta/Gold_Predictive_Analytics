# Laporan Proyek Machine Learning - Hanif Al Irsyad
---

## Domain Proyek
Domain yang dipilih untuk proyek machine learning ini adalah Keuangan, dengan judul Predictive Analytics : Predictive Analytics Gold Price

### Latar Belakang
Pada dasarnya ada dua jenis pasar saham yang ada, yaitu pasar ekuitas dan pasar komoditas. Pasar ekuitas adalah agregasi dari produsen dan konsumen saham dan perdagangan primer selain produk manufaktur adalah pasar komoditas. pasar komoditas. Ada dua jenis komoditas yang ada di pasar komoditas, salah satunya adalah komoditas lunak di di mana gandum, kopi, kakao dan gula datang dan lainnya adalah komoditas keras di mana emas, karet dan minyak datang.

Orang yang berinvestasi emas terutama memiliki dua tujuan utama, salah satunya adalah berlindung terhadap inflasi selama periode waktu tertentu, pengembalian investasi emas sejalan dengan tingkat inflasi. Berinvestasi dalam emas telah berkembang selama periode waktu tertentu untuk cara tradisional dengan membeli perhiasan atau dengan cara modern seperti membeli koin emas dan batangan (yang tersedia di bank terjadwal) atau dengan berinvestasi di Gold Dana yang diperdagangkan di bursa (ETF Emas).

Pergerakan harga emas yang cukup signifikan dan besarnya keuntungan yang ditawarkan, ternyata menarik banyak masyarakat untuk berkecimpung dalam forex market. Forex market memiliki fungsi pokok dalam membantu kelancaran lalu lintas pembayaran internasional.

Forex termasuk investasi kategori high risk dengan kata lain beresiko tinggi karena transaksi yang kurang tepat sasarang dapat langsung mengikis modal deposito dalam sebuah akun dengan cepat. Oleh karena itu, para trader harus mengetahui kapan harus masuk, berapa lama menunggu dan berapa kali harus melakukan pembelian/penjualan. Salah satu cara yang dapat dilakukan adalah dengan menggunakan teknik *forecasting*.

*Forecasting* adalah suatu teknik untuk meramalkan keadaan dimasa yang akan datang dengan menggunakan data-data yang telah ada di masa lalu. Hal ini termasuk dalam *time series forecasting*, dengan mendeteksi pola dan kecenderungan data *time series* kemudian memformulasikannya dalam suatu model, maka dapat digunakan untuk memprediksi data yang akan datang.

## Business Understanding
---

### Problem Statement
Berdasarkan pada latar belakang di atas, permasalahan yang dapat diselesaikan pada proyek ini adalah sebagai berikut :
* Bagaimana menganalisa data harga *Emas*?
* Bagaimana cara mengolah data agar di latih dengan baik oleh model?
* Bagaimana cara membangun model yang dapat memprediksi *time series forecasting* dengan baik?

### Goals
Tujuan proyek ini dibuat adalah sebagai berikut :
* Dapat memprediksi harga *Emas* dengan akurat menggunakan model machine learning.
* Melakukan analisa dan mengolah data yang optimal agar diterima dengan baik oleh model machine learning.
* Membantu para *trader* dalam melakukan pembelian pada *Emas*.

### Solution Statement
Solusi yang dapat dilakukan agar goals terpenuhi adalah sebagai berikut :
* Melakukan analisa, eksplorasi, pemrosesan pada data dengan memvisualisasikan data agar mendapat gambaran bagaimana data tersebut. Berikut adalah analisa yang dapat dilakukan :
    * Menangani *missing value* pada data.
    * Mencari korelasi pada data untuk mencari *dependant variable* dan *independent variable*.
    * Menangani outlier pada data dengan menggunakan Metode IQR.
    * Melakukan normalisasi pada data terutama pada fitur numerik.
    * Membuat model regresi untuk memprediksi bilangan kontinu untuk memprediksi harga yang akan datang. 
    
* Berikut beberapa algoritma yang digunakan pada proyek ini :
    * Support Vector Machine (Support Vector Regression)
    * K-Nearest Neighbors
    * Boosting Algorithm (Gradient Boosting Regression)
    
* Melakukan hyperparameter tuning agar model dapat berjalan pada performa terbaik dengan menggunakan teknik Grid Search

## Data Understanding
---

Dataset yang digunakan pada proyek ini adalah dataset dari Kaggle yang berjudul *Commodity Futures Price History* [https://www.kaggle.com/datasets/mattiuzc/commodity-futures-price-history]

Dataset yang digunakan memiliki format *.csv* yang mempunyai total 5291 data dengan 7 kolom (*Date, Open, High, Low, Close, Adj Close, Volume*) yang memiliki total 112 *missing value* pada masing-masing kolom *Open, High, Low, Close, Adj Close, Volume* dengan informasi sebagai berikut :
  * Date : Tanggal pencatatan Data
  * Open : Harga buka dihitung perhari
  * High : Harga tertinggi perhari
  * Low : Harga terendah perhari
  * Close : Harga tutup dihitung perhari
  * Adj Close : Harga penutupan pada hari tersebut setelah disesuaikan dengan aksi korporasi seperti right issue, stock split atau stock reverse.
  * Volume : Volume transaksi

### Exploratory Data Analysis
Sebelum melakukan pemrosesan data, kita harus mengetahui keadaan data. seperti mencari korelasi antar fitur, mencari outlier, melakukan analisis *univariate* dan *multivariate*.

* Menangani outlier
Jika data numerik divisualisasikan, hanya fitur *Volume* saja yang memiliki outlier. Untuk menangani outlier kita akan menggunakan IQR Method yaitu dengan menghapus data yang berada diluar IQR yaitu antara 25% dan 75%. setelah melakukan kegiatan mengatasi outlier, didapatkan sampel 4550 Data dan 7 Kolom.

* Univariate Analysis
Pada kasus ini kita hanya akan berfokus dalam memprediksi *Adj Close*.

* Multivariate Analysis
Selanjutnya kita akan menganalisis korelasi fitur *Adj Close* terhadap fitur lain seperti *Open, High, Low, Close dan Volume*. Dapat disimpulkan bahwa *Adj Close* memiliki korelasi positif yang kuat terhadap *Open, High, Low dan Close*, sedangkan untuk fitur *Volume* memiliki korelasi sedang terhadap fitur *Adj Close*.
    
Untuk memperjelas korelasi kita akan memvisualisasikannya menggunakan heatmap dari library Seaborn. Dapat kita lihat bahwa *Adj Close* memiliki korelasi positif tinggi pada setiap fitur, kecuali fitur *Volume* sehingga kita dapat menggunakan semua fitur sebagai *dependant variable*. 
    
# Data Preparation
---

Berikut ini merupakan tahapan-tahapan dalam melakukan pra-pemrosesan data:
### Melakukan Penanganan Missing Value
Pada kasus ini dalam menangani Missing Value menggunakan library SimpleImputer, yang dimana library ini bertugas untuk mengisi kolom yang memiliki missing value dengan data mean (nilai rata-rata)

### Melakukan pembagian dataset
Kita akan membagi dataset menjadi 2 yaitu sebagai train data dan test data. Train data digunakan sebagai training model dan test data digunakan sebagai validasi apakah model sudah akurat atau belum. Ratio yang umum dalam splitting dataset adalah 80:20, 80% sebagai train data dan 20% sebagai test data, sehingga kita akan menggunakan Ratio tersebut. Pembagian dataset dilakukan dengan modul train_test_split dari scikit-learn. Setelah melakukan pembagian dataset, didapatkan jumlah sample pada data latih yaitu 3640 sampel dan jumlah sample pada data test yaitu 910 sampel dari total jumlah sample pada dataset yaitu 4550 sampel.
    
### Menghapus fitur yang tidak diperlukan
Karena kita tidak memerlukan fitur *Date* dan *Volume* kita akan menghapus fitur *Date* dan *Volume*. Juga kita tidak memerlukan fitur *Close* karena *Adj Close* lebih akurat dari pada *Close* sehingga kita menghapus fitur *Close*.

### Data Normalization
Normalisasi data digunakan agar model dapat bekerja lebih optimal karena model tidak perlu mengolah data dengan angka besar. Normalisasi biasanya mentransformasi data dalam skala tertentu. Untuk proyek ini kita akan normalisasi data 0 hingga 1 menggunakan MinMaxScaler.

# Modeling
---

Model yang akan digunakan proyek kali ini yaitu *Support Vector Regression, Gradient Boosting,* dan *K-Nearest Neighbors*.

### Support Vector Regression
*Support Vector Regression* memiliki prinsip yang sama dengan SVM, namun SVM biasa digunakan dalam klasifikasi. Pada SVM, algoritma tersebut berusaha mencari jalan terbesar yang bisa memisahkan sampel dari kelas berbeda, sedangkan SVR mencari jalan yang dapat menampung sebanyak mungkin sampel di jalan. Untuk hyper parameter yang digunakan pada model ini adalah sebagai berikut :
* *kernel* : Hyperparameter ini digunakan untuk menghitung kernel matriks sebelumnya.
* *C* : Hyperparameter ini adalah parameter regularisasi digunakan untuk menukar klasifikasi yang benar dari contoh *training* terhadap maksimalisasi margin fungsi keputusan.
* *gamma* : Hyperparameter ini digunakan untk menetukan seberapa jauh pengaruh satu contoh pelatihan mencapai, dengan nilai rendah berarti jauh dan nilai tinggi berarti dekat.

##### Kelebihan
* Lebih efektif pada data dimensi tinggi (data dengan jumlah fitur yang banyak)
* Memori lebih efisien karena menggunakan subset poin pelatihan

##### Kekurangan
* Sulit dipakai pada data skala besar

### K-Nearest Neighbors
*K-Nearest Neighbors* merupakan algoritma machine learning yang bekerja dengan mengklasifikasikan data baru menggunakan kemiripan antara data baru dengan sejumlah data (k) pada data yang telah ada. Algoritma ini dapat digunakan untuk klasifikasi dan regresi. Untuk hyperparameter yang digunakan pada model ini hanya 1 yaitu :
* *n_neighbors* : Jumlah tetangga untuk yang diperlukan untuk menentukan letak data baru

##### Kelebihan
* Dapat menerima data yang masih *noisy*
* Sangat efektif apabila jumlah datanya banyak
* Mudah diimplementasikan

##### Kekurangan
* Sensitif pada outlier
* Rentan pada fitur yang kurang informatif

### Gradient Boosting
Gradient Boosting adalah algoritma machine learning yang menggunakan teknik *ensembel learning* dari *decision tree* untuk memprediksi nilai. Gradient Boosting sangat mampu menangani pattern yang kompleks dan data ketika linear model tidak dapat menangani. Untuk hyperparameter yang digunakan pada model ini ada 3 yaitu :
* *learning_rate* : Hyperparameter training yang digunakan untuk menghitung nilai koreksi bobot padad waktu proses training. Umumnya nilai learning rate berkisar antara 0 hingga 1
* *n_estimators* : Jumlah tahapan boosting yang akan dilakukan.
* *criterion* : Hyperparameter yang digunakan untuk menemukan fitur dan ambang batas optimal dalam membagi data

##### Kelebihan
* Hasil pemodelan yang lebih akurat
* Model yang stabil dan lebih kuat (robust)
* Dapat digunakan untuk menangkap hubungan linear maupun non linear pada data

##### Kekurangan
* Pengurangan kemampuan interpretasi model
* Waktu komputasi dan desain tinggi
* Tingkat kesulitan yang tinggi dalam pemilihan model

Untuk proyek kali ini kita akan menggunakan model *K-Nearest Neighbors* karena memiliki error (*0.00001*) yang paling sedikit daripada model yang lain. Namun tidak bisa dipungkiri model dari Gradient Boosting juga memiliki error (*0.000011*) yang hampir seperti *KNN*.

# Evaluation
---

Untuk evaluasi pada machine learning model ini, metrik yang digunakan adalah *mean squared error (mse)*. Dimana metrik ini mengukur seberapa dekat garis pas dengan titik data.

![](https://www.gstatic.com/education/formulas2/443397389/en/mean_squared_error.svg)

dimana :
n = jumlah titik data
Yi = nilai sesungguhnya
Yi_hat = nilai prediksi

Untuk proyek kali ini terdapat 2 model yang dapat berjalan dengan performa optimal yaitu, *Gradient Boosting* model dan *K-Nearest Neighbors*. Sehingga dapat disimpulkan bahwa model dapat memprediksi harga dari pasar *foreign exchange (forex)* dari data tes dengan baik. Sehingga kedepannya dapat membantu para *trader* dalam melakukan keputusan pembelian/penjualan pasar.

# Referensi :
---

* Navin,Dr. G. Vadivu. (2015). Big Data Analytics for Gold Price Forecasting Based on Decision Tree Algorithm and Support Vector Regression (SVR). Retrieved March, 2015, from https://www.researchgate.net/profile/Dr-Vadivu-G/publication/274238769_Big_Data_Analytics_for_Gold_Price_Forecasting_Based_on_Decision_Tree_Algorithm_and_Support_Vector_Regression_SVR/links/555ac15c08ae6fd2d8283610/Big-Data-Analytics-for-Gold-Price-Forecasting-Based-on-Decision-Tree-Algorithm-and-Support-Vector-Regression-SVR.pdf
* Saputri, L. (2016). IMPLEMENTASI JARINGAN SARAF TIRUAN RADIAL BASIS FUNCTION (RBF) PADA PERAMALAN FOREIGN EXCHANGE (FOREX). 
* Foreign Exchange turnover in April 2019. The Bank for International Settlements. (2019, September 16). Retrieved August 20, 2022, from https://www.bis.org/statistics/rpfx19_fx.htm 
* Hussein, S., &amp; About The AuthorSaddam HusseinTerlibat di pekerjaan dunia geospasial sejak 2011. Sekarang bekerja di sebuah NGO internasional yang bergerak di bidang konservasi satwa liar. (2022, February 2). Ensemble learning dalam machine learning: Bagging Dan Boosting. GEOSPASIALIS. Retrieved August 22, 2022, from https://geospasialis.com/ensemble-learning/ 
* Aliyev, V. (2020, October 7). Gradient boosting classification explained through python. Medium. Retrieved August 22, 2022, from https://towardsdatascience.com/gradient-boosting-classification-explained-through-python-60cc980eeb3d 
