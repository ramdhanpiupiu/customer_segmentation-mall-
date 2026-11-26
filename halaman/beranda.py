import streamlit as st

from helper import FITUR_NUM, catatan, fmt, hero, langkah, muat_data, statistik, tim

df = muat_data()
hero('Tugas Machine Learning · Clustering',
     'Customer Segmentation (Mall)',
     'Siapa saja pelanggan yang punya pola pendapatan dan kebiasaan belanja yang mirip? '
     'Proyek ini menjawabnya dengan segmentasi K-Means pada 200 pelanggan.',
     ['EDA', 'PCA', 'K-Means', 'Streamlit'])

pf = (df['Gender'] == 'Female').mean() * 100
statistik([('Jumlah pelanggan', len(df)),
           ('Fitur numerik', len(FITUR_NUM)),
           ('Data kosong', int(df.isna().sum().sum())),
           ('Female / Male', f'{fmt(pf, 0)}% / {fmt(100 - pf, 0)}%')])

st.subheader('Anggota Kelompok')
tim()

st.subheader('Gambaran singkat')
st.write('Aplikasi ini adalah rangkuman pengerjaan segmentasi pelanggan, dari mengenal data sampai memakai model '
         'untuk menebak segmen pelanggan baru. Setiap angka dan grafik dihitung ulang dari berkas '
         '`Mall_Customers.csv` setiap kali halaman dibuka.')

st.subheader('Pertanyaan yang ingin dijawab')
st.markdown('1. Seperti apa karakteristik pelanggan pada data ini?\n'
            '2. Berapa banyak segmen yang paling masuk akal?\n'
            '3. Algoritma clustering mana yang menghasilkan kelompok paling rapi?\n'
            '4. Bagaimana menentukan segmen untuk pelanggan baru?')

catatan('Perlu diingat',
        'Data ini <b>tidak punya kolom jawaban</b> (label), jadi tidak ada kelompok yang "benar". '
        'Model hanya mencari kemiripan, lalu kualitasnya dinilai dengan ukuran kerapian kelompok '
        'seperti <b>silhouette score</b>, bukan accuracy.')

st.subheader('Jalur pengerjaan')
langkah([('Kenali data', 'Latar belakang, sumber, dan isi data'),
         ('Jelajahi', 'Sebaran, korelasi, dan outlier'),
         ('Siapkan', 'Cek kualitas, encoding, standardisasi'),
         ('Ringkas', 'PCA untuk melihat struktur data'),
         ('Tentukan K', 'Elbow dan silhouette'),
         ('Modelkan', 'Bandingkan tiga algoritma'),
         ('Gunakan', 'Cek satu atau banyak pelanggan'),
         ('Simpulkan', 'Hasil, keterbatasan, dan saran')])

with st.expander('Cara memakai aplikasi ini'):
    st.markdown('- Gunakan menu di sisi kiri untuk berpindah halaman, dari atas ke bawah sesuai jalur pengerjaan.\n'
                '- Pada halaman **Optimasi Jumlah Segmen (K)**, tombol "Gunakan K ini" mengubah K yang dipakai halaman '
                'lain (modelling, prediksi pelanggan, dan kesimpulan).\n'
                '- Halaman **Prediksi Massal (CSV)** menerima CSV dengan kolom Annual Income (k$) dan Spending Score (1-100).')

with st.expander('Istilah yang dipakai'):
    st.markdown('- **Clustering**: mengelompokkan data berdasarkan kemiripan tanpa label.\n'
                '- **K-Means**: algoritma yang membagi data ke K kelompok dengan titik pusat masing-masing.\n'
                '- **Standardisasi**: menyamakan skala fitur (rata-rata 0, simpangan baku 1).\n'
                '- **PCA**: meringkas banyak fitur menjadi beberapa komponen utama.\n'
                '- **Inertia**: jumlah jarak kuadrat titik ke pusat kelompoknya (makin kecil makin rapat).\n'
                '- **Silhouette score**: ukuran kerapian kelompok dari -1 sampai 1 (makin besar makin baik).')
