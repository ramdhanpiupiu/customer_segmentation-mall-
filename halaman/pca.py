import altair as alt
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from helper import FITUR_SEMUA, fmt, judul_halaman, matriks, muat_data

df = muat_data()
judul_halaman('Reduksi Dimensi (PCA)', 'Meringkas empat fitur menjadi komponen utama agar struktur data terlihat.')

x = StandardScaler().fit_transform(matriks(df, FITUR_SEMUA))
pca = PCA().fit(x)
var = pca.explained_variance_ratio_

st.subheader('Seberapa banyak informasi tertangkap')
tabel = pd.DataFrame({'Komponen': [f'PC{i + 1}' for i in range(len(var))],
                      'Varians (%)': (var * 100).round(2), 'Kumulatif (%)': (var.cumsum() * 100).round(2)})
st.dataframe(tabel, hide_index=True, width='stretch')
st.altair_chart(alt.Chart(tabel).mark_bar().encode(x='Komponen', y='Varians (%)'), width='stretch')
n = st.slider('Jumlah komponen yang dipertahankan', 1, 4, 2)
st.success(f'{n} komponen menangkap {fmt(var[:n].sum() * 100)}% varians data.')

st.subheader('Kontribusi tiap fitur')
st.dataframe(pd.DataFrame(pca.components_[:2].T, index=FITUR_SEMUA, columns=['PC1', 'PC2']).round(3), width='stretch')

st.subheader('Peta pelanggan dalam 2 dimensi')
proj = pd.DataFrame(pca.transform(x)[:, :2], columns=['PC1', 'PC2'])
proj['Gender'] = df['Gender'].values
st.altair_chart(alt.Chart(proj).mark_circle(size=60).encode(
    x='PC1', y='PC2', color='Gender', tooltip=['PC1', 'PC2', 'Gender']).interactive(), width='stretch')
st.caption('PCA dipakai untuk memahami struktur data. Model final tetap memakai fitur asli agar mudah ditafsirkan.')
