import pandas as pd
import streamlit as st

from helper import FITUR_MODEL, ambil_k, judul_halaman, model_final, muat_data

judul_halaman('Prediksi Massal (CSV)', 'Tentukan segmen banyak pelanggan sekaligus lewat file CSV.')
k = ambil_k()
scaler, model, _, _ = model_final(k)

st.write(f'CSV wajib punya kolom `{FITUR_MODEL[0]}` dan `{FITUR_MODEL[1]}` (model memakai K = {k}).')
unggah = st.file_uploader('Pilih file CSV', type='csv')
contoh = st.checkbox('Pakai Mall_Customers.csv sebagai contoh')

data = None
if unggah is not None:
    data = pd.read_csv(unggah)
elif contoh:
    data = muat_data()

if data is None:
    st.info('Pilih file CSV atau centang contoh untuk memulai.')
    st.stop()

kurang = [c for c in FITUR_MODEL if c not in data.columns]
if kurang:
    st.error(f'Kolom tidak ditemukan: {kurang}')
    st.stop()

valid = data.dropna(subset=FITUR_MODEL).copy()
if len(valid) < len(data):
    st.warning(f'{len(data) - len(valid)} baris berisi data kosong dan dilewati.')
valid['Segmen'] = model.predict(scaler.transform(valid[FITUR_MODEL]))

c1, c2 = st.columns(2)
c1.metric('Pelanggan diproses', len(valid))
c2.metric('Segmen yang terisi', valid['Segmen'].nunique())
st.write('Jumlah pelanggan per segmen')
st.bar_chart(valid['Segmen'].value_counts().sort_index())
st.dataframe(valid, width='stretch')
st.download_button('Unduh hasil (CSV)', valid.to_csv(index=False).encode('utf-8'),
                   file_name='hasil_segmentasi.csv', mime='text/csv')
