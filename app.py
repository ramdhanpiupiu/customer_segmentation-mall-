import streamlit as st

from helper import PENULIS, terapkan_css

st.set_page_config(page_title='Segmentasi Pelanggan Mall', layout='wide')
terapkan_css()

menu = {
    'Pengantar': [
        st.Page('halaman/beranda.py', title='Beranda', icon=':material/home:', default=True),
        st.Page('halaman/latar_data.py', title='Tentang Dataset', icon=':material/database:'),
        st.Page('halaman/eda.py', title='Eksplorasi Data (EDA)', icon=':material/bar_chart:'),
    ],
    'Pengolahan & Model': [
        st.Page('halaman/persiapan.py', title='Preprocessing', icon=':material/tune:'),
        st.Page('halaman/pca.py', title='Reduksi Dimensi (PCA)', icon=':material/bubble_chart:'),
        st.Page('halaman/jumlah_segmen.py', title='Optimasi Jumlah Segmen (K)', icon=':material/auto_fix_high:'),
        st.Page('halaman/pemodelan.py', title='Modelling & Evaluasi', icon=':material/model_training:'),
    ],
    'Penerapan': [
        st.Page('halaman/cek_satu.py', title='Prediksi Satu Pelanggan', icon=':material/person_search:'),
        st.Page('halaman/cek_banyak.py', title='Prediksi Massal (CSV)', icon=':material/upload_file:'),
        st.Page('halaman/kesimpulan.py', title='Kesimpulan & Saran', icon=':material/task_alt:'),
    ],
}
halaman = st.navigation(menu)

with st.sidebar:
    st.divider()
    st.caption(' segmentasi pelanggan · data Mall Customers')
    st.markdown('**Anggota Kelompok**')
    for nama, nim in PENULIS:
        st.caption(f'{nama} · {nim}')

halaman.run()
