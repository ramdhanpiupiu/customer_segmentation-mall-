import streamlit as st

from helper import ambil_k, fmt, judul_halaman, model_final, profil_segmen

judul_halaman('Kesimpulan & Saran', 'Ringkasan temuan, keterbatasan, dan arah pengembangan.')
k = ambil_k()
_, _, tersegmen, sil = model_final(k)
profil = profil_segmen(tersegmen)

st.subheader('Kesimpulan')
st.markdown(f'1. Data berisi **{len(tersegmen)} pelanggan** tanpa data kosong dan tanpa label, sehingga persoalannya '
            'adalah clustering.\n'
            f'2. Model final: **K-Means dengan K = {k}** pada Annual Income dan Spending Score yang distandarkan, '
            f'dengan silhouette score **{fmt(sil, 3)}**.\n'
            '3. Setiap segmen punya karakter yang terbaca dari rata-rata Income dan Spending (tabel di bawah).\n'
            '4. Model dapat menentukan segmen pelanggan baru, satu per satu maupun lewat CSV.')
st.dataframe(profil, width='stretch')

st.subheader('Keterbatasan')
st.markdown('- Dataset hanya 200 pelanggan dan dibuat untuk pembelajaran, sehingga hasilnya ilustratif.\n'
            '- Tanpa label, kebenaran segmen hanya dapat dinilai lewat metrik kerapian kelompok.\n'
            '- Interpretasi segmen bersifat umum dan perlu dikonfirmasi pihak yang memahami bisnis mall.')

st.subheader('Saran')
st.markdown('1. Uji ulang dengan data pelanggan yang lebih besar dan lebih beragam.\n'
            '2. Tambahkan fitur lain, misalnya frekuensi kunjungan atau kategori belanja, bila datanya tersedia.\n'
            '3. Coba algoritma lain seperti DBSCAN untuk mendeteksi pelanggan yang tidak masuk kelompok mana pun.')
