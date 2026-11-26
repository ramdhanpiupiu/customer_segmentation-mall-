import altair as alt
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from helper import (KOL_INCOME, KOL_SKOR, SET_FITUR, ambil_k, fmt, judul_halaman, matriks, model_final,
                    muat_data, profil_segmen)


@st.cache_data
def bandingkan(fitur, k):
    x = StandardScaler().fit_transform(matriks(muat_data(), fitur))
    algo = {'K-Means': KMeans(n_clusters=k, n_init=10, random_state=42),
            'Agglomerative (Ward)': AgglomerativeClustering(n_clusters=k),
            'Gaussian Mixture': GaussianMixture(n_components=k, random_state=42)}
    baris = []
    for nama, a in algo.items():
        label = a.fit_predict(x)
        baris.append({'Algoritma': nama, 'Silhouette': silhouette_score(x, label),
                      'Davies-Bouldin': davies_bouldin_score(x, label),
                      'Calinski-Harabasz': calinski_harabasz_score(x, label)})
    return pd.DataFrame(baris)


judul_halaman('Modelling & Evaluasi', 'Membandingkan tiga algoritma clustering lalu memeriksa model final.')
k = st.slider('Jumlah segmen (K)', 2, 10, ambil_k())
nama_set = st.selectbox('Fitur yang dipakai', list(SET_FITUR), index=0)

st.subheader('Perbandingan algoritma')
hasil = bandingkan(SET_FITUR[nama_set], k)
st.dataframe(hasil.round(3), hide_index=True, width='stretch')
st.caption('Silhouette dan Calinski-Harabasz: makin besar makin baik. Davies-Bouldin: makin kecil makin baik. '
           'Skor antar himpunan fitur dengan jumlah dimensi berbeda tidak sepenuhnya sebanding.')
terbaik = hasil.loc[hasil['Silhouette'].idxmax(), 'Algoritma']
st.success(f'Silhouette tertinggi pada pengaturan ini: {terbaik}.')

st.subheader(f'Model final: K-Means pada Income dan Spending (K = {k})')
st.write('K-Means dipilih sebagai model final karena pusat tiap segmen mudah dibaca, dan model dapat langsung '
         'dipakai untuk menebak segmen pelanggan baru. Agglomerative tidak memiliki fungsi prediksi untuk data baru.')
scaler, model, tersegmen, sil = model_final(k)
st.metric('Silhouette score', fmt(sil, 3))
pusat = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=[KOL_INCOME, KOL_SKOR])
pusat['Segmen'] = range(k)
titik = alt.Chart(tersegmen).mark_circle(size=60, opacity=.7).encode(
    x=KOL_INCOME, y=KOL_SKOR, color='Segmen:N',
    tooltip=['CustomerID', 'Gender', 'Age', KOL_INCOME, KOL_SKOR, 'Segmen'])
tengah = alt.Chart(pusat).mark_point(shape='diamond', size=250, filled=True, color='black').encode(
    x=KOL_INCOME, y=KOL_SKOR)
st.altair_chart((titik + tengah).interactive(), width='stretch')
st.write('Profil tiap segmen (rata-rata)')
st.dataframe(profil_segmen(tersegmen), width='stretch')
