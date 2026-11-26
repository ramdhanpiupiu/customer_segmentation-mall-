import altair as alt
import streamlit as st

from helper import SET_FITUR, ambil_k, evaluasi_k, fmt, judul_halaman, k_terbaik

judul_halaman('Optimasi Jumlah Segmen (K)', 'Memilih K dengan metode Elbow dan Silhouette pada K-Means.')

nama_set = st.selectbox('Fitur yang dievaluasi', list(SET_FITUR), index=0)
ev = evaluasi_k(SET_FITUR[nama_set])

kiri, kanan = st.columns(2)
with kiri:
    st.write('Elbow: inertia (makin kecil makin rapat)')
    st.altair_chart(alt.Chart(ev).mark_line(point=True).encode(x='K:O', y='Inertia'), width='stretch')
with kanan:
    st.write('Silhouette score (makin besar makin baik)')
    st.altair_chart(alt.Chart(ev).mark_line(point=True).encode(x='K:O', y='Silhouette'), width='stretch')

st.dataframe(ev.round(3), hide_index=True, width='stretch')
terbaik = int(ev.loc[ev['Silhouette'].idxmax(), 'K'])
st.success(f"Silhouette tertinggi ada pada K = {terbaik} (skor {fmt(ev['Silhouette'].max(), 3)}).")

st.subheader('Tetapkan K yang dipakai')
st.write(f'K saat ini: **{ambil_k()}** (rekomendasi untuk Income + Spending: {k_terbaik()}).')
k_pilih = st.number_input('Pilih K', min_value=2, max_value=10, value=terbaik, step=1)
if st.button('Gunakan K ini'):
    st.session_state['k_final'] = int(k_pilih)
    st.rerun()
