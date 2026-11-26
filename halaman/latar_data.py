import altair as alt
import pandas as pd
import streamlit as st

from helper import (KOL_INCOME, KOL_SKOR, KOL_UMUR, URL_KAGGLE, FITUR_NUM, catatan, fmt,
                    judul_halaman, muat_data, statistik)

df = muat_data()
judul_halaman('Tentang Dataset', 'Mengapa segmentasi pelanggan, dari mana data berasal, dan apa isinya.')

tab1, tab2, tab3, tab4 = st.tabs(['Latar belakang', 'Sumber data', 'Struktur data', 'Kualitas & keterbatasan'])

with tab1:
    st.subheader('Latar belakang')
    st.write('Pusat perbelanjaan dikunjungi orang-orang dengan kemampuan finansial dan kebiasaan belanja yang '
             'berbeda-beda. Memperlakukan semua pengunjung dengan cara yang sama membuat promosi dan program '
             'pelanggan sulit tepat sasaran. Segmentasi pelanggan membagi pengunjung ke dalam kelompok yang '
             'karakternya mirip, sehingga setiap kelompok dapat dipahami secara terpisah.')
    st.write('Data yang dipakai berisi 200 pelanggan dengan atribut umur, gender, pendapatan tahunan, dan skor '
             'belanja. Karena tidak ada kolom yang menyatakan kelompok "benar", pendekatan yang sesuai adalah '
             '**clustering**, yaitu bagian dari unsupervised learning.')
    st.subheader('Rumusan masalah')
    st.markdown('1. Bagaimana karakteristik pelanggan pada dataset Mall Customers?\n'
                '2. Berapa jumlah segmen yang paling sesuai?\n'
                '3. Algoritma clustering mana yang paling baik berdasarkan metrik evaluasi?\n'
                '4. Bagaimana menentukan segmen pelanggan baru dengan model yang dipilih?')
    k1, k2 = st.columns(2)
    with k1:
        st.subheader('Tujuan')
        st.markdown('- Mengenali dan menjelajahi data pelanggan.\n- Menyiapkan data untuk clustering.\n'
                    '- Menentukan jumlah segmen dan membandingkan algoritma.\n'
                    '- Membuat aplikasi web interaktif untuk memakai model.')
    with k2:
        st.subheader('Batasan')
        st.markdown('- Model final memakai Annual Income dan Spending Score.\n- Jumlah segmen diuji pada K = 2 sampai 10.\n'
                    '- Algoritma yang dibandingkan: K-Means, Agglomerative, Gaussian Mixture.\n'
                    '- Data tidak memuat waktu, lokasi, maupun transaksi.')

with tab2:
    st.subheader('Sumber data')
    info = pd.DataFrame({
        'Keterangan': ['Nama dataset', 'Penyedia', 'Pengunggah', 'Berkas', 'Format dan ukuran',
                       'Skenario data', 'Tujuan pembuatan'],
        'Isi': ['Mall Customers (Customer Segmentation Tutorial in Python)', 'Kaggle', 'Vijay Choudhary (vjchoudhary7)',
                'Mall_Customers.csv', f'CSV, {df.shape[0]} baris x {df.shape[1]} kolom',
                'Pelanggan pemegang kartu anggota sebuah pusat perbelanjaan',
                'Untuk belajar konsep segmentasi pelanggan']})
    st.dataframe(info, hide_index=True, width='stretch')
    st.link_button('Buka halaman dataset di Kaggle', URL_KAGGLE)
    catatan('Catatan penting tentang sumber data',
            'Skenario dan tujuan pembuatan di atas mengikuti keterangan pada halaman Kaggle yang dikutip berbagai '
            'sumber sekunder. Dataset ini dibuat untuk <b>pembelajaran</b>, bukan data transaksi mall yang terverifikasi. '
            'Karena itu, hasil segmentasi pada aplikasi ini bersifat <b>ilustratif</b> dan tidak boleh dianggap '
            'cerminan pelanggan mall yang sebenarnya.')
    st.subheader('Cara mengutip')
    st.code('Choudhary, V. Customer Segmentation Tutorial in Python [Data set]. Kaggle. '
            + URL_KAGGLE, language='text')

with tab3:
    statistik([('Jumlah baris', df.shape[0]), ('Jumlah kolom', df.shape[1]),
               ('Fitur numerik', len(FITUR_NUM)), ('Kolom kategorik', 1)])
    ket = {'CustomerID': 'Nomor identitas pelanggan (pengenal, bukan fitur)',
           'Gender': 'Jenis kelamin pelanggan',
           'Age': 'Umur pelanggan (tahun)',
           'Annual Income (k$)': 'Pendapatan tahunan dalam ribuan dolar',
           'Spending Score (1-100)': 'Skor belanja yang diberikan mall (1 sampai 100)'}
    rentang = []
    for c in df.columns:
        if c == 'Gender':
            rentang.append(', '.join(f'{k} ({v})' for k, v in df[c].value_counts().items()))
        else:
            rentang.append(f'{df[c].min()} sampai {df[c].max()}')
    st.dataframe(pd.DataFrame({'Atribut': df.columns, 'Tipe data': df.dtypes.astype(str).values,
                               'Rentang / kategori': rentang, 'Keterangan': [ket[c] for c in df.columns]}),
                 hide_index=True, width='stretch')
    n = st.slider('Jumlah baris pratinjau', 5, 50, 10)
    st.dataframe(df.head(n), width='stretch')
    st.write('Statistik deskriptif')
    st.dataframe(df.drop(columns='CustomerID').describe().round(2), width='stretch')

with tab4:
    st.subheader('Kualitas data')
    statistik([('Data kosong', int(df.isna().sum().sum())),
               ('Baris duplikat', int(df.duplicated().sum())),
               ('ID unik', 'Ya' if df['CustomerID'].is_unique else 'Tidak'),
               ('Rata-rata skor belanja', fmt(df[KOL_SKOR].mean(), 1))])
    c1, c2 = st.columns(2)
    with c1:
        st.write('Komposisi Gender')
        st.bar_chart(df['Gender'].value_counts())
    with c2:
        st.write('Sebaran umur')
        st.altair_chart(alt.Chart(df).mark_bar().encode(x=alt.X(KOL_UMUR, bin=alt.Bin(maxbins=12)), y='count()'),
                        width='stretch')
    st.subheader('Keterbatasan')
    st.markdown('- Hanya 200 pelanggan, sehingga hasil belum tentu berlaku pada populasi lebih besar.\n'
                '- Cara penghitungan Spending Score ditentukan penyedia data dan tidak dijelaskan rinci.\n'
                '- Tidak ada label kelompok, sehingga hasil clustering tidak bisa dicek terhadap jawaban benar.\n'
                '- Gender hanya terdiri dari dua kategori dan tidak ada atribut waktu atau lokasi.\n'
                '- Dataset dibuat untuk pembelajaran, bukan data operasional mall.')
