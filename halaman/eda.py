import altair as alt
import pandas as pd
import streamlit as st

from helper import FITUR_NUM, KOL_INCOME, KOL_SKOR, KOL_UMUR, fmt, judul_halaman, matriks, muat_data

df = muat_data()
judul_halaman('Eksplorasi Data (EDA)', 'Melihat sebaran dan hubungan antar fitur sebelum membangun model.')

st.subheader('Sebaran fitur')
kolom = st.radio('Fitur yang dilihat', FITUR_NUM, horizontal=True)
st.altair_chart(alt.Chart(df).mark_bar().encode(
    x=alt.X(kolom, bin=alt.Bin(maxbins=20)), y='count()', color='Gender'), width='stretch')

st.subheader('Female dibanding Male')
st.altair_chart(alt.Chart(df).mark_boxplot(extent=1.5).encode(x='Gender', y=kolom, color='Gender'), width='stretch')

st.subheader('Berdasarkan kelompok umur')
batas = [17, 25, 35, 45, 55, 100]
nama = ['18-25', '26-35', '36-45', '46-55', '56+']
dfu = df.assign(Kelompok_umur=pd.cut(df[KOL_UMUR], bins=batas, labels=nama))
ringkas = dfu.groupby('Kelompok_umur', observed=True).agg(
    Jumlah=('CustomerID', 'count'), Income=(KOL_INCOME, 'mean'), Spending=(KOL_SKOR, 'mean')).round(1)
u1, u2 = st.columns(2)
u1.write('Rata-rata Spending Score')
u1.bar_chart(ringkas['Spending'])
u2.write('Ringkasan per kelompok umur')
u2.dataframe(ringkas, width='stretch')

st.subheader('Korelasi antar fitur')
korelasi = matriks(df, FITUR_NUM + ['Gender_enc']).corr().round(2)
long = korelasi.reset_index().melt('index')
long.columns = ['Fitur A', 'Fitur B', 'Korelasi']
heat = alt.Chart(long).mark_rect().encode(
    x='Fitur A:N', y='Fitur B:N', color=alt.Color('Korelasi:Q', scale=alt.Scale(scheme='tealblues', domain=[-1, 1])))
teks = alt.Chart(long).mark_text().encode(x='Fitur A:N', y='Fitur B:N', text='Korelasi:Q')
st.altair_chart(heat + teks, width='stretch')
st.caption('Gender_enc: Female = 0, Male = 1.')

st.subheader('Income terhadap Spending Score')
st.altair_chart(alt.Chart(df).mark_circle(size=70).encode(
    x=KOL_INCOME, y=KOL_SKOR, color='Gender',
    tooltip=['CustomerID', 'Gender', KOL_UMUR, KOL_INCOME, KOL_SKOR]).interactive(), width='stretch')

st.subheader('Pemeriksaan outlier (aturan IQR)')
baris = []
for c in FITUR_NUM:
    q1, q3 = df[c].quantile(.25), df[c].quantile(.75)
    iqr = q3 - q1
    bb, ba = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    baris.append({'Fitur': c, 'Batas bawah': round(bb, 1), 'Batas atas': round(ba, 1),
                  'Jumlah outlier': int(((df[c] < bb) | (df[c] > ba)).sum())})
st.table(baris)

st.subheader('Temuan')
st.markdown(f"- Korelasi Age dengan Spending Score: **{fmt(korelasi.loc[KOL_UMUR, KOL_SKOR])}**.\n"
            f"- Korelasi Annual Income dengan Spending Score: **{fmt(korelasi.loc[KOL_INCOME, KOL_SKOR])}**, "
            'sehingga hubungan liniernya lemah dan pola kelompok perlu dicari dengan clustering.\n'
            '- Jumlah outlier tiap fitur ada pada tabel IQR di atas.')
