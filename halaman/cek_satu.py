import altair as alt
import pandas as pd
import streamlit as st

from helper import (KOL_INCOME, KOL_SKOR, ambil_k, judul_halaman, model_final, muat_data, prediksi_segmen,
                    profil_segmen)

judul_halaman('Prediksi Satu Pelanggan', 'Masukkan data seorang pelanggan untuk melihat segmennya.')
k = ambil_k()
_, _, tersegmen, _ = model_final(k)
profil = profil_segmen(tersegmen)
rata_skor = muat_data()[KOL_SKOR].mean()

with st.form('form_satu'):
    a, b = st.columns(2)
    with a:
        nama = st.text_input('Nama pelanggan')
        gender = st.selectbox('Gender', ['Female', 'Male'])
        umur = st.number_input('Umur', min_value=18, max_value=100, value=30, step=1)
    with b:
        income = st.number_input('Annual Income (k$)', min_value=0, max_value=300, value=60, step=1)
        skor = st.number_input('Spending Score (1-100)', min_value=1, max_value=100, value=50, step=1)
    kirim = st.form_submit_button('Lihat segmen')
st.caption(f'Model (K = {k}) hanya memakai Annual Income dan Spending Score. Umur dan Gender hanya dicatat.')

if kirim:
    seg = prediksi_segmen(k, income, skor)
    st.success(f'{nama or "Pelanggan"} ({gender}, {umur} th) masuk **Segmen {seg}**: {profil.loc[seg, "Karakter"]}.')
    m1, m2, m3 = st.columns(3)
    m1.metric('Segmen', seg)
    m2.metric('Anggota segmen', int(profil.loc[seg, 'Jumlah']))
    m3.metric('Selisih skor vs rata-rata', f'{skor - rata_skor:+.1f}')
    st.info(f"Interpretasi (ilustratif): {profil.loc[seg, 'Interpretasi (ilustratif)']}")
    st.dataframe(profil.loc[[seg]], width='stretch')
    baru = pd.DataFrame({KOL_INCOME: [income], KOL_SKOR: [skor]})
    dasar = alt.Chart(tersegmen).mark_circle(size=50, opacity=.5).encode(x=KOL_INCOME, y=KOL_SKOR, color='Segmen:N')
    sorot = alt.Chart(baru).mark_point(shape='diamond', size=300, filled=True, color='black').encode(
        x=KOL_INCOME, y=KOL_SKOR)
    st.altair_chart(dasar + sorot, width='stretch')
