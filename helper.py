from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / 'Mall_Customers.csv'
KOL_UMUR = 'Age'
KOL_INCOME = 'Annual Income (k$)'
KOL_SKOR = 'Spending Score (1-100)'
FITUR_NUM = [KOL_UMUR, KOL_INCOME, KOL_SKOR]
FITUR_MODEL = [KOL_INCOME, KOL_SKOR]
FITUR_SEMUA = [KOL_UMUR, KOL_INCOME, KOL_SKOR, 'Gender_enc']
SET_FITUR = {
    'Income + Spending (2 fitur)': (KOL_INCOME, KOL_SKOR),
    'Umur + Income + Spending (3 fitur)': (KOL_UMUR, KOL_INCOME, KOL_SKOR),
    'Semua fitur + Gender (4 fitur)': tuple(FITUR_SEMUA),
}
PENULIS = [
    ('Aditya Pratama Rahawarin', '241011402075'),
    ('Raden Mas Radithia Akmal', '241011400696'),
    ('Rifqi Ramdhan', '241011401857'),
]

URL_KAGGLE = 'https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python'

SARAN_SEGMEN = {
    ('tinggi', 'tinggi'): 'Pelanggan berdaya beli tinggi yang aktif berbelanja. Cocok dipertahankan lewat program loyalitas.',
    ('tinggi', 'rendah'): 'Berdaya beli tinggi tetapi belanjanya rendah. Berpotensi digarap lewat penawaran yang lebih relevan.',
    ('rendah', 'tinggi'): 'Berdaya beli lebih rendah tetapi aktif berbelanja. Perlu dipantau agar belanja tetap terjangkau.',
    ('rendah', 'rendah'): 'Berdaya beli dan belanja sama-sama rendah. Prioritas pendekatan relatif rendah.',
}

CSS = """
<style>
.hero{background:linear-gradient(135deg,#0b3c49 0%,#145c6b 55%,#1c8c8c 100%);border-radius:14px;
      padding:2rem 2.2rem;color:#fff;margin-bottom:1.2rem}
.hero .kicker{color:#ffd166;text-transform:uppercase;letter-spacing:.14em;font-size:.74rem;font-weight:700}
.hero h1{color:#fff;margin:.4rem 0 .6rem 0;padding:0;font-size:2.2rem;line-height:1.2}
.hero p{color:#e6f4f1;margin:0;font-size:1rem;max-width:46rem}
.hero .chips span{display:inline-block;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.3);
      border-radius:999px;padding:.15rem .8rem;margin:.9rem .4rem 0 0;font-size:.8rem}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.9rem;margin:.6rem 0 1rem}
.stat{border-radius:12px;padding:.9rem 1rem;background:rgba(15,157,138,.09);border-top:3px solid #0f9d8a}
.stat .t{font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;opacity:.8;font-weight:600}
.stat .n{font-size:1.9rem;font-weight:700;line-height:1.3}
.note{background:rgba(255,193,7,.13);border:1px dashed #d9a400;border-radius:12px;padding:.9rem 1.2rem;margin:1rem 0}
.note b.judul{display:block;margin-bottom:.2rem}
.steps{display:flex;flex-wrap:wrap;gap:.9rem;margin:.6rem 0 1rem}
.step{flex:1 1 150px;text-align:center;padding:.4rem}
.step i{display:inline-block;width:2.1rem;height:2.1rem;line-height:2.1rem;border-radius:50%;background:#0f9d8a;
      color:#fff;font-style:normal;font-weight:700;margin-bottom:.4rem}
.step b{display:block}
.step span{font-size:.85rem;opacity:.8}
.tim-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.8rem;margin:.4rem 0 1rem}
.anggota{border-radius:12px;padding:.7rem 1rem;border:1px solid rgba(128,128,128,.3)}
.anggota b{display:block}
.anggota span{font-size:.82rem;opacity:.75}
</style>
"""


def fmt(x, d=2):
    s = f'{x:,.{d}f}'
    return s.replace(',', 'X').replace('.', ',').replace('X', '.')


def terapkan_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(kicker, judul, subjudul, chips):
    c = ''.join(f'<span>{x}</span>' for x in chips)
    st.markdown(f'<div class="hero"><div class="kicker">{kicker}</div><h1>{judul}</h1>'
                f'<p>{subjudul}</p><div class="chips">{c}</div></div>', unsafe_allow_html=True)


def statistik(items):
    isi = ''.join(f'<div class="stat"><div class="t">{l}</div><div class="n">{v}</div></div>' for l, v in items)
    st.markdown(f'<div class="stat-grid">{isi}</div>', unsafe_allow_html=True)


def catatan(judul, isi):
    st.markdown(f'<div class="note"><b class="judul">{judul}</b>{isi}</div>', unsafe_allow_html=True)


def langkah(items):
    isi = ''.join(f'<div class="step"><i>{n}</i><b>{j}</b><span>{d}</span></div>'
                  for n, (j, d) in enumerate(items, 1))
    st.markdown(f'<div class="steps">{isi}</div>', unsafe_allow_html=True)


def tim():
    isi = ''.join(f'<div class="anggota"><b>{n}</b><span>NIM {nim}</span></div>' for n, nim in PENULIS)
    st.markdown(f'<div class="tim-grid">{isi}</div>', unsafe_allow_html=True)


def judul_halaman(judul, deskripsi):
    st.title(judul)
    st.caption(deskripsi)


@st.cache_data
def muat_data():
    return pd.read_csv(DATA_PATH)


def matriks(df, fitur):
    x = df.copy()
    x['Gender_enc'] = (x['Gender'] == 'Male').astype(int)
    return x[list(fitur)]


@st.cache_data
def evaluasi_k(fitur, k_min=2, k_max=10):
    x = StandardScaler().fit_transform(matriks(muat_data(), fitur))
    baris = []
    for k in range(k_min, k_max + 1):
        m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(x)
        baris.append({'K': k, 'Inertia': m.inertia_, 'Silhouette': silhouette_score(x, m.labels_)})
    return pd.DataFrame(baris)


@st.cache_data
def k_terbaik():
    ev = evaluasi_k(tuple(FITUR_MODEL))
    return int(ev.loc[ev['Silhouette'].idxmax(), 'K'])


def ambil_k():
    return st.session_state.get('k_final') or k_terbaik()


@st.cache_resource
def model_final(k):
    df = muat_data()
    scaler = StandardScaler().fit(df[FITUR_MODEL])
    x = scaler.transform(df[FITUR_MODEL])
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(x)
    hasil = df.copy()
    hasil['Segmen'] = model.labels_
    return scaler, model, hasil, float(silhouette_score(x, model.labels_))


def profil_segmen(hasil):
    p = hasil.groupby('Segmen').agg(
        Jumlah=('CustomerID', 'count'), Umur=(KOL_UMUR, 'mean'),
        Income=(KOL_INCOME, 'mean'), Spending=(KOL_SKOR, 'mean'),
        Persen_Female=('Gender', lambda s: (s == 'Female').mean() * 100)).round(1)
    mi, ms = hasil[KOL_INCOME].mean(), hasil[KOL_SKOR].mean()
    inc = ['tinggi' if v > mi else 'rendah' for v in p['Income']]
    spd = ['tinggi' if v > ms else 'rendah' for v in p['Spending']]
    p['Karakter'] = [f'Income {a}, Spending {b}' for a, b in zip(inc, spd)]
    p['Interpretasi (ilustratif)'] = [SARAN_SEGMEN[(a, b)] for a, b in zip(inc, spd)]
    return p.rename(columns={'Persen_Female': '% Female'})


def prediksi_segmen(k, income, skor):
    scaler, model, _, _ = model_final(k)
    x = scaler.transform(pd.DataFrame([[income, skor]], columns=FITUR_MODEL))
    return int(model.predict(x)[0])
