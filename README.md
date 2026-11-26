# Dashboard Kualitas Udara DKI Jakarta

Dashboard Streamlit untuk menganalisis ISPU (Indeks Standar Pencemar Udara) dan pengaruh meteorologi di DKI Jakarta, 2024-2025.

## Isi dashboard
1. **Tren ISPU**: perubahan ISPU dari waktu ke waktu (harian, mingguan, bulanan) dan pola musiman
2. **Antarstasiun**: perbandingan ISPU, kategori, dan parameter kritis antar 5 stasiun
3. **Pengaruh Cuaca**: korelasi suhu, kelembapan, angin, hujan, dan radiasi matahari dengan ISPU
4. **Clustering**: K-Means untuk mengelompokkan stasiun berdasarkan karakter pencemar
5. **Klasifikasi**: Random Forest untuk memprediksi kategori kualitas udara

## Struktur
```
app.py            tampilan dashboard
analisis.py       fungsi analisis (korelasi, clustering, klasifikasi)
prepare_data.py   membersihkan file mentah .xls menjadi CSV
data/             CSV bersih yang dipakai dashboard
.streamlit/       pengaturan tema warna
```

## Menjalankan di komputer sendiri
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Membuat ulang data dari file mentah
Taruh file `.xls` mentah di folder `raw/`, lalu jalankan `python prepare_data.py`.

## Sumber data
Data ISPU dan Data Meteorologi Provinsi DKI Jakarta (Satu Data Jakarta).
