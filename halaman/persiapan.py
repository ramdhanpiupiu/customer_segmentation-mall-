import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from helper import FITUR_NUM, judul_halaman, matriks, muat_data

df = muat_data()
judul_halaman('Preprocessing', 'Menyiapkan data agar siap dipakai algoritma clustering.')

st.subheader('1. Cek kualitas')
a, b = st.columns(2)
a.write('Data kosong per kolom')
a.dataframe(df.isna().sum().rename('Jumlah'), width='stretch')
b.metric('Baris duplikat (semua kolom)', int(df.duplicated().sum()))
b.metric('Baris duplikat (tanpa CustomerID)', int(df.drop(columns='CustomerID').duplicated().sum()))

st.subheader('2. Memilih fitur')
st.write('`CustomerID` hanya nomor pengenal dan tidak membawa informasi kemiripan, jadi tidak dipakai. '
         'Fitur yang tersedia: `Age`, `Annual Income (k$)`, `Spending Score (1-100)`, dan `Gender`.')

st.subheader('3. Mengubah Gender menjadi angka')
st.write('Clustering bekerja dengan angka, sehingga **Female = 0** dan **Male = 1**.')
contoh = matriks(df, ['Age', 'Gender_enc']).assign(Gender=df['Gender'])[['Gender', 'Gender_enc', 'Age']].head(8)
st.dataframe(contoh, width='stretch')

st.subheader('4. Menyamakan skala')
st.write('K-Means bekerja dengan jarak, sehingga fitur berskala besar bisa mendominasi. Standardisasi membuat tiap '
         'fitur memiliki rata-rata 0 dan simpangan baku 1.')
pilihan = st.radio('Tampilkan data', ['Sebelum standardisasi', 'Sesudah standardisasi'], horizontal=True)
x = df[FITUR_NUM]
if pilihan == 'Sesudah standardisasi':
    x = pd.DataFrame(StandardScaler().fit_transform(x), columns=FITUR_NUM)
st.dataframe(x.describe().loc[['mean', 'std', 'min', 'max']].round(2), width='stretch')

st.subheader('5. Outlier')
st.write('Pemeriksaan outlier ada pada halaman Eksplorasi Data. Tidak ada data yang dihapus agar seluruh '
         '200 pelanggan tetap mendapat segmen.')
