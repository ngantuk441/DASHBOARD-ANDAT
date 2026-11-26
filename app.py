import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analisis import (
    CUACA, LABEL_CUACA, LABEL_POLUTAN, POLUTAN, URUTAN_KATEGORI,
    fitur_klaster, gabung, jalankan_klaster, korelasi, latih_klasifikasi,
    muat_data, tabel_silhouette,
)

st.set_page_config(page_title="Dashboard Kualitas Udara DKI Jakarta", page_icon="🌫️", layout="wide")

# ------------------------------------------------------------------ palet warna
P = ["#F0F3FA", "#D5DEEF", "#B1C9EF", "#8AAEE0", "#628ECB", "#395886"]
GELAP = "#1F3358"  # turunan lebih gelap dari #395886 untuk teks & kontras
SKALA = [[i / 5, c] for i, c in enumerate(P)]
WARNA_KAT = {"BAIK": P[2], "SEDANG": P[4], "TIDAK SEHAT": P[5], "SANGAT TIDAK SEHAT": GELAP}
WARNA_STASIUN = {
    "DKI1 Bundaran HI": P[3], "DKI2 Kelapa Gading": P[4], "DKI3 Jagakarsa": P[5],
    "DKI4 Lubang Buaya": GELAP, "DKI5 Kebon Jeruk": P[2],
    "Rata-rata semua stasiun": GELAP,
}
WARNA_KLASTER = {f"Klaster {i + 1}": c for i, c in enumerate([P[3], P[5], P[2], P[4], GELAP, P[1]])}
NAMA_PARAM = {"PM25": "PM2.5", "PM10": "PM10", "SO2": "SO₂", "O3": "O₃", "NO2": "NO₂", "CO": "CO"}
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

st.markdown(
    f"""
<style>
[data-testid="stSidebar"] {{background: linear-gradient(180deg, {P[5]} 0%, #2A4468 100%);}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label p, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p
{{color: {P[0]} !important;}}
[data-testid="stMetric"] {{background:#FFFFFF; border:1px solid {P[1]}; border-radius:14px; padding:14px 18px;}}
.banner {{background: linear-gradient(120deg, {P[5]} 0%, {P[4]} 70%, {P[3]} 100%);
  padding: 28px 34px; border-radius: 18px; color: {P[0]}; margin-bottom: 18px;}}
.banner h1 {{color:{P[0]}; margin:0; font-size:2rem; padding:0;}}
.banner p {{margin:6px 0 0 0; font-size:1.05rem; color:{P[1]};}}
</style>
""",
    unsafe_allow_html=True,
)


def banner(judul, sub):
    st.markdown(f'<div class="banner"><h1>{judul}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def gaya(fig, tinggi=420):
    fig.update_layout(
        height=tinggi, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        font=dict(color=GELAP), margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.18, title_text=""),
    )
    fig.update_xaxes(gridcolor=P[1], zeroline=False)
    fig.update_yaxes(gridcolor=P[1], zeroline=False)
    return fig


@st.cache_data
def muat():
    return muat_data("data")


ispu_all, meteo_all = muat()

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("## Menu Dashboard")
HALAMAN = [
    "Beranda",
    "1. Tren ISPU",
    "2. Antarstasiun",
    "3. Pengaruh Cuaca",
    "4. Clustering Stasiun",
    "5. Klasifikasi Kategori",
]
halaman = st.sidebar.radio("Pilih Analisis:", HALAMAN)

st.sidebar.markdown("## Filter")
semua_stasiun = sorted(ispu_all["stasiun"].unique())
pilih_stasiun = st.sidebar.multiselect("Stasiun:", semua_stasiun, default=semua_stasiun)
tmin, tmax = ispu_all["tanggal"].min().date(), ispu_all["tanggal"].max().date()
rentang = st.sidebar.date_input("Periode:", value=(tmin, tmax), min_value=tmin, max_value=tmax)
if isinstance(rentang, (tuple, list)) and len(rentang) == 2:
    t0, t1 = rentang
else:
    t0, t1 = tmin, tmax

st.sidebar.markdown("---")
st.sidebar.caption(
    "Sumber: Data ISPU & Data Meteorologi Provinsi DKI Jakarta (Satu Data Jakarta). "
    "ISPU = Indeks Standar Pencemar Udara."
)

mask = (
    ispu_all["stasiun"].isin(pilih_stasiun)
    & (ispu_all["tanggal"].dt.date >= t0)
    & (ispu_all["tanggal"].dt.date <= t1)
)
df = ispu_all[mask].copy()
if df.empty:
    st.warning("Tidak ada data untuk filter yang dipilih. Ubah stasiun atau periode di sidebar.")
    st.stop()

# ================================================================== BERANDA
if halaman == "Beranda":
    banner("Kualitas Udara DKI Jakarta", "Analisis ISPU dan pengaruh kondisi meteorologi, 2024 – 2025")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah observasi harian", f"{len(df):,}".replace(",", "."))
    c2.metric("Stasiun pemantauan", df["stasiun"].nunique())
    c3.metric("Rata-rata ISPU", f"{df['ispu'].mean():.1f}")
    c4.metric("Hari tidak sehat", f"{df['kategori'].isin(['TIDAK SEHAT', 'SANGAT TIDAK SEHAT']).mean() * 100:.1f}%")

    st.subheader("Pertanyaan yang dijawab dashboard ini")
    st.markdown(
        """
1. **Tren ISPU**: bagaimana kualitas udara berubah dari waktu ke waktu?
2. **Antarstasiun**: bagaimana perbedaan kualitas udara antar stasiun pemantauan?
3. **Pengaruh cuaca**: bagaimana hubungan suhu, kelembapan, angin, hujan, dan radiasi matahari dengan ISPU?
4. **Clustering**: stasiun (atau stasiun-bulan) mana yang punya karakter kualitas udara mirip?
5. **Klasifikasi**: seberapa baik kategori kualitas udara bisa ditebak dari nilai parameter pencemar?
"""
    )
    with st.expander("Catatan data dan pembersihan"):
        st.markdown(
            """
- **Skala ISPU**: Baik 0–50, Sedang 51–100, Tidak Sehat 101–200, Sangat Tidak Sehat 201–300.
  Nilai ISPU harian adalah nilai **tertinggi** dari sub-indeks tiap parameter (PM10, PM2.5, SO₂, CO, O₃, NO₂).
- Nama stasiun di data mentah tidak seragam (mis. *Bunderan HI* dan *Bundaran Hotel Indonesia (HI)*), jadi disatukan menjadi 5 stasiun.
- Baris dengan kategori *Tidak ada data* dibuang. Data cuaca per 10 menit dirata-ratakan menjadi harian (curah hujan dijumlahkan).
- Nilai sensor mustahil (mis. suhu −40 °C atau angin 360 m/s) dibuang sebelum dirata-ratakan.
- Data cuaca dan ISPU digabung hanya untuk tanggal dan stasiun yang sama-sama tersedia.
"""
        )

# ================================================================== 1. TREN
elif halaman == HALAMAN[1]:
    banner("Perubahan ISPU dari Waktu ke Waktu", "Bagaimana kualitas udara Jakarta berubah sepanjang periode pengamatan?")
    k1, k2, k3, k4 = st.columns(4)
    tdk = df["kategori"].isin(["TIDAK SEHAT", "SANGAT TIDAK SEHAT"])
    k1.metric("Rata-rata ISPU", f"{df['ispu'].mean():.1f}")
    k2.metric("Hari kategori Baik", f"{(df['kategori'] == 'BAIK').mean() * 100:.1f}%")
    k3.metric("Hari Tidak Sehat", f"{tdk.mean() * 100:.1f}%")
    top = df.loc[df["ispu"].idxmax()]
    k4.metric("ISPU tertinggi", f"{top['ispu']:.0f}", f"{top['tanggal']:%d %b %Y}", delta_color="off")

    c1, c2 = st.columns([3, 1])
    agregasi = c1.radio("Agregasi waktu", ["Harian", "Mingguan", "Bulanan"], index=2, horizontal=True)
    gab = c2.checkbox("Rata-ratakan semua stasiun")
    freq = {"Harian": "D", "Mingguan": "W", "Bulanan": "MS"}[agregasi]
    idx = df.set_index("tanggal")
    if gab:
        tren = idx["ispu"].resample(freq).mean().reset_index()
        tren["stasiun"] = "Rata-rata semua stasiun"
    else:
        tren = idx.groupby("stasiun")["ispu"].resample(freq).mean().reset_index()
    fig = px.line(tren, x="tanggal", y="ispu", color="stasiun", color_discrete_map=WARNA_STASIUN,
                  title=f"Nilai ISPU ({agregasi.lower()})", labels={"tanggal": "", "ispu": "ISPU"})
    fig.add_hline(y=50, line_dash="dot", line_color=P[3], annotation_text="Batas Baik (50)")
    fig.add_hline(y=100, line_dash="dot", line_color=P[5], annotation_text="Batas Sedang (100)")
    st.plotly_chart(gaya(fig, 440), use_container_width=True)

    d = df.copy()
    d["bulan"] = d["tanggal"].dt.to_period("M").dt.to_timestamp()
    komp = d.groupby(["bulan", "kategori"]).size().reset_index(name="n")
    komp["persen"] = komp["n"] / komp.groupby("bulan")["n"].transform("sum") * 100
    fig = px.bar(komp, x="bulan", y="persen", color="kategori", color_discrete_map=WARNA_KAT,
                 category_orders={"kategori": URUTAN_KATEGORI}, title="Komposisi kategori kualitas udara per bulan",
                 labels={"bulan": "", "persen": "% hari"})
    st.plotly_chart(gaya(fig, 380), use_container_width=True)

    d["bln"] = d["tanggal"].dt.month
    d["tahun"] = d["tanggal"].dt.year.astype(str)
    musim = d.groupby(["tahun", "bln"])["ispu"].mean().reset_index()
    fig = px.line(musim, x="bln", y="ispu", color="tahun", markers=True,
                  color_discrete_sequence=[P[3], P[5]], title="Pola musiman: rata-rata ISPU per bulan",
                  labels={"bln": "", "ispu": "ISPU"})
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=BULAN)
    st.plotly_chart(gaya(fig, 360), use_container_width=True)

    rata_bln = d.groupby("bln")["ispu"].mean()
    st.info(
        f"Dari seluruh data terpilih, bulan dengan rata-rata ISPU **tertinggi** adalah "
        f"**{BULAN[rata_bln.idxmax() - 1]}** ({rata_bln.max():.1f}) dan **terendah** adalah "
        f"**{BULAN[rata_bln.idxmin() - 1]}** ({rata_bln.min():.1f})."
    )

# ================================================================== 2. ANTARSTASIUN
elif halaman == HALAMAN[2]:
    banner("Perbandingan Antarstasiun", "Stasiun mana yang kualitas udaranya paling baik dan paling buruk?")
    ring = df.groupby("stasiun").agg(
        rata_rata=("ispu", "mean"), median=("ispu", "median"), maks=("ispu", "max"),
        persen_tidak_sehat=("kategori", lambda s: s.isin(["TIDAK SEHAT", "SANGAT TIDAK SEHAT"]).mean() * 100),
        n=("ispu", "size"),
    ).round(1).sort_values("rata_rata")
    st.dataframe(
        ring.rename(columns={"rata_rata": "Rata-rata ISPU", "median": "Median", "maks": "Maksimum",
                             "persen_tidak_sehat": "% Hari Tidak Sehat", "n": "Jumlah hari"}),
        use_container_width=True,
    )
    c1, c2 = st.columns(2)
    fig = px.box(df, x="stasiun", y="ispu", color="stasiun", color_discrete_map=WARNA_STASIUN, points=False,
                 title="Sebaran nilai ISPU", labels={"stasiun": "", "ispu": "ISPU"})
    fig.update_layout(showlegend=False)
    c1.plotly_chart(gaya(fig, 400), use_container_width=True)

    kom = df.groupby(["stasiun", "kategori"]).size().reset_index(name="n")
    kom["persen"] = kom["n"] / kom.groupby("stasiun")["n"].transform("sum") * 100
    fig = px.bar(kom, y="stasiun", x="persen", color="kategori", orientation="h", color_discrete_map=WARNA_KAT,
                 category_orders={"kategori": URUTAN_KATEGORI}, title="Komposisi kategori per stasiun",
                 labels={"stasiun": "", "persen": "% hari"})
    c2.plotly_chart(gaya(fig, 400), use_container_width=True)

    hm = df.groupby("stasiun")[POLUTAN].mean().rename(columns=LABEL_POLUTAN)
    fig = px.imshow(hm, text_auto=".1f", aspect="auto", color_continuous_scale=SKALA,
                    title="Rata-rata sub-indeks ISPU per parameter",
                    labels={"x": "Parameter", "y": "", "color": "Nilai"})
    st.plotly_chart(gaya(fig, 340), use_container_width=True)

    kr = df.dropna(subset=["parameter_kritis"]).copy()
    kr["parameter"] = kr["parameter_kritis"].map(NAMA_PARAM).fillna(kr["parameter_kritis"])
    kr = kr.groupby(["stasiun", "parameter"]).size().reset_index(name="n")
    kr["persen"] = kr["n"] / kr.groupby("stasiun")["n"].transform("sum") * 100
    fig = px.bar(kr, x="stasiun", y="persen", color="parameter", barmode="group",
                 color_discrete_sequence=[P[2], P[3], P[4], P[5], GELAP, P[1]],
                 title="Parameter pencemar kritis (penentu ISPU harian)", labels={"stasiun": "", "persen": "% hari"})
    st.plotly_chart(gaya(fig, 380), use_container_width=True)

    terbaik, terburuk = ring.index[0], ring.index[-1]
    st.info(
        f"Rata-rata ISPU **terendah (udara relatif paling baik)**: **{terbaik}** ({ring.loc[terbaik, 'rata_rata']}). "
        f"**Tertinggi**: **{terburuk}** ({ring.loc[terburuk, 'rata_rata']})."
    )

# ================================================================== 3. CUACA
elif halaman == HALAMAN[3]:
    banner("Hubungan Cuaca dengan ISPU", "Suhu, kelembapan, kecepatan angin, curah hujan, dan radiasi matahari")
    g = gabung(df, meteo_all)
    if len(g) < 30:
        st.warning("Data gabungan ISPU dan cuaca terlalu sedikit untuk filter ini.")
        st.stop()
    st.caption(f"Menggunakan {len(g):,} observasi harian (stasiun-tanggal) yang punya data ISPU dan cuaca.".replace(",", "."))

    cor = korelasi(g)
    nama = {**LABEL_CUACA, "ispu": "ISPU"}
    cor_n = cor.rename(index=nama, columns=nama)
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Heatmap(
        z=cor_n.abs().values, x=list(cor_n.columns), y=list(cor_n.index), colorscale=SKALA, zmin=0, zmax=1,
        text=cor_n.round(2).values, texttemplate="%{text}", showscale=False,
    ))
    fig.update_layout(title="Korelasi Spearman antar variabel (warna = kekuatan)")
    fig.update_yaxes(autorange="reversed")
    c1.plotly_chart(gaya(fig, 400), use_container_width=True)

    r = cor["ispu"].drop("ispu").sort_values()
    fig = go.Figure(go.Bar(x=r.values, y=[LABEL_CUACA[k] for k in r.index], orientation="h", marker_color=P[4],
                           text=[f"{v:+.2f}" for v in r.values], textposition="outside"))
    fig.update_layout(title="Korelasi tiap variabel cuaca dengan ISPU")
    fig.update_xaxes(range=[-0.7, 0.7], title="Koefisien Spearman")
    c2.plotly_chart(gaya(fig, 400), use_container_width=True)

    def kekuatan(v):
        a = abs(v)
        return "sangat lemah" if a < 0.1 else "lemah" if a < 0.3 else "sedang" if a < 0.5 else "kuat"

    tabel = pd.DataFrame({
        "Variabel": [LABEL_CUACA[k] for k in r.index[::-1]],
        "Koefisien": [round(v, 2) for v in r.values[::-1]],
        "Arah": ["searah (naik bersama ISPU)" if v > 0 else "berlawanan (naik saat ISPU turun)" for v in r.values[::-1]],
        "Kekuatan": [kekuatan(v) for v in r.values[::-1]],
    })
    st.dataframe(tabel, use_container_width=True, hide_index=True)

    st.subheader("Lihat satu variabel lebih dekat")
    var = st.selectbox("Variabel cuaca", CUACA, format_func=lambda k: LABEL_CUACA[k])
    s = g.dropna(subset=[var, "ispu"])
    samp = s.sample(min(len(s), 2500), random_state=1)
    fig = px.scatter(samp, x=var, y="ispu", color="stasiun", color_discrete_map=WARNA_STASIUN, opacity=0.55,
                     labels={var: LABEL_CUACA[var], "ispu": "ISPU"}, title=f"ISPU vs {LABEL_CUACA[var]}")
    if s[var].nunique() > 1:
        a, b = np.polyfit(s[var], s["ispu"], 1)
        xs = np.linspace(s[var].min(), s[var].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=a * xs + b, mode="lines", name="Garis tren", line=dict(color=GELAP, width=3)))
    st.plotly_chart(gaya(fig, 420), use_container_width=True)

    g["kondisi"] = np.where(g["hujan"] > 1, "Hari hujan (> 1 mm)", "Tidak hujan / gerimis")
    fig = px.box(g, x="kondisi", y="ispu", color="kondisi", points=False,
                 color_discrete_map={"Hari hujan (> 1 mm)": P[3], "Tidak hujan / gerimis": P[5]},
                 title="ISPU pada hari hujan vs tidak hujan", labels={"kondisi": "", "ispu": "ISPU"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(gaya(fig, 360), use_container_width=True)
    st.warning("Korelasi menunjukkan keterkaitan, bukan sebab-akibat. Nilai |r| di bawah 0,3 berarti hubungan lemah.")

# ================================================================== 4. CLUSTERING
elif halaman == HALAMAN[4]:
    banner("Clustering Stasiun Pemantauan", "Pengelompokan berdasarkan karakteristik parameter pencemar (K-Means)")
    c1, c2 = st.columns(2)
    level = c1.radio("Unit yang dikelompokkan", ["Stasiun-Bulan", "Stasiun"], horizontal=True,
                     help="'Stasiun' hanya 1 titik per stasiun (maks. 5 titik). 'Stasiun-Bulan' = rata-rata per stasiun per bulan, lebih kaya.")
    agg = fitur_klaster(df, level)
    if len(agg) < 4:
        st.warning("Data terlalu sedikit untuk clustering. Pilih lebih banyak stasiun atau periode lebih panjang.")
        st.stop()
    k_maks = min(6, len(agg) - 1)
    k = c2.slider("Jumlah klaster (k)", 2, k_maks, min(3, k_maks))
    hasil, sil, var = jalankan_klaster(agg, k)

    m1, m2, m3 = st.columns(3)
    m1.metric("Jumlah titik data", len(hasil))
    m2.metric("Silhouette score", f"{sil:.2f}", help="Makin mendekati 1 makin terpisah jelas. Di bawah 0,25 berarti klaster saling tumpang tindih.")
    m3.metric("Varians dijelaskan 2 sumbu", f"{var.sum() * 100:.0f}%")

    sil_tab = tabel_silhouette(agg)
    ca, cb = st.columns(2)
    fig = px.scatter(hasil, x="PC1", y="PC2", color="klaster", symbol="stasiun", color_discrete_map=WARNA_KLASTER,
                     hover_data={"bulan": True, "ispu": ":.1f", "PC1": False, "PC2": False},
                     category_orders={"klaster": sorted(hasil["klaster"].unique())},
                     title="Peta klaster (PCA 2 dimensi)",
                     labels={"PC1": f"PC1 ({var[0] * 100:.0f}%)", "PC2": f"PC2 ({var[1] * 100:.0f}%)"})
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color=GELAP)))
    ca.plotly_chart(gaya(fig, 430), use_container_width=True)

    fig = px.bar(sil_tab, x="k", y="silhouette", title="Silhouette score untuk berbagai k", labels={"silhouette": "Silhouette", "k": "Jumlah klaster"})
    fig.update_traces(marker_color=P[4])
    fig.update_xaxes(dtick=1)
    cb.plotly_chart(gaya(fig, 430), use_container_width=True)

    prof = hasil.groupby("klaster")[POLUTAN + ["ispu"]].mean().rename(columns={**LABEL_POLUTAN, "ispu": "ISPU rata-rata"})
    fig = px.imshow(prof, text_auto=".1f", aspect="auto", color_continuous_scale=SKALA,
                    title="Profil tiap klaster (rata-rata sub-indeks)", labels={"x": "", "y": "", "color": "Nilai"})
    st.plotly_chart(gaya(fig, 300), use_container_width=True)

    ct = hasil.groupby(["stasiun", "klaster"]).size().reset_index(name="n")
    fig = px.bar(ct, x="stasiun", y="n", color="klaster", color_discrete_map=WARNA_KLASTER,
                 title="Isi tiap klaster berdasarkan stasiun", labels={"stasiun": "", "n": "Jumlah titik"})
    st.plotly_chart(gaya(fig, 360), use_container_width=True)

    # ringkasan otomatis: parameter yang paling menonjol di tiap klaster
    rata_all = hasil[POLUTAN].mean()
    sd_all = hasil[POLUTAN].std().replace(0, np.nan)
    baris = []
    for kl, sub in hasil.groupby("klaster"):
        z = ((sub[POLUTAN].mean() - rata_all) / sd_all).dropna().sort_values(ascending=False)
        unggul = LABEL_POLUTAN[z.index[0]] if len(z) else "-"
        stas = sub["stasiun"].value_counts().index[0]
        baris.append(f"- **{kl}**: {len(sub)} titik, ISPU rata-rata {sub['ispu'].mean():.1f}, paling menonjol di **{unggul}**, paling banyak berasal dari {stas}.")
    st.markdown("\n".join(baris))
    if sil < 0.25:
        st.warning("Silhouette rendah: pemisahan antar klaster tidak tegas, jadi baca klaster sebagai kecenderungan, bukan batas yang pasti.")

# ================================================================== 5. KLASIFIKASI
else:
    banner("Klasifikasi Kategori Kualitas Udara", "Memprediksi kategori ISPU dari nilai parameter pencemar (Random Forest)")
    st.info(
        "Kategori ISPU ditentukan dari nilai sub-indeks **tertinggi** di antara parameter. Karena itu model yang memakai "
        "parameter ini hampir pasti akurat; nilai pentingnya ada pada **parameter mana yang paling menentukan kategori**."
    )
    c1, c2 = st.columns(2)
    uji = c1.slider("Porsi data uji", 0.10, 0.40, 0.25, 0.05)
    pakai_stasiun = c2.checkbox("Sertakan stasiun sebagai fitur")
    if len(df) < 150:
        st.warning("Data terlalu sedikit untuk melatih model. Pilih lebih banyak stasiun atau periode lebih panjang.")
        st.stop()

    @st.cache_data(show_spinner="Melatih model...")
    def latih(data, uji, pakai_stasiun):
        return latih_klasifikasi(data, ukuran_uji=uji, pakai_stasiun=pakai_stasiun)

    hasil = latih(df, uji, pakai_stasiun)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Akurasi", f"{hasil['akurasi'] * 100:.1f}%")
    m2.metric("F1 makro", f"{hasil['f1_makro']:.3f}", help="Rata-rata F1 semua kelas, adil untuk kelas yang jarang.")
    m3.metric("Data latih", hasil["n_latih"])
    m4.metric("Data uji", hasil["n_uji"])

    ca, cb = st.columns(2)
    fig = px.imshow(hasil["cm"], x=hasil["kelas"], y=hasil["kelas"], text_auto=True, color_continuous_scale=SKALA,
                    labels={"x": "Prediksi", "y": "Aktual", "color": "Jumlah"}, title="Confusion matrix (data uji)")
    ca.plotly_chart(gaya(fig, 380), use_container_width=True)

    pent = hasil["penting"].sort_values()
    nama_f = [LABEL_POLUTAN.get(i, i.replace("stasiun_", "")) for i in pent.index]
    fig = go.Figure(go.Bar(x=pent.values, y=nama_f, orientation="h", marker_color=P[4]))
    fig.update_layout(title="Tingkat kepentingan fitur")
    cb.plotly_chart(gaya(fig, 380), use_container_width=True)

    ps = hasil["per_stasiun"].copy()
    ps["akurasi"] = (ps["akurasi"] * 100).round(1)
    st.subheader("Akurasi per stasiun (data uji)")
    st.dataframe(ps.rename(columns={"stasiun": "Stasiun", "akurasi": "Akurasi (%)", "n_uji": "Jumlah data uji"}),
                 use_container_width=True, hide_index=True)
    st.caption("Kelas 'Sangat Tidak Sehat' hanya 1 kejadian di data, jadi digabung ke 'Tidak Sehat' agar model bisa dievaluasi.")

    st.subheader("Coba prediksi sendiri")
    st.caption("Isi nilai sub-indeks tiap parameter, lalu lihat kategori yang diprediksi.")
    med = df[POLUTAN].median()
    cols = st.columns(6)
    nilai = {p: cols[i].number_input(LABEL_POLUTAN[p], 0, 300, int(med[p]), key=f"in_{p}") for i, p in enumerate(POLUTAN)}
    baris = {p: nilai[p] for p in POLUTAN}
    if pakai_stasiun:
        st_pilih = st.selectbox("Stasiun", sorted(df["stasiun"].unique()))
        for kol in hasil["kolom"]:
            if kol.startswith("stasiun_"):
                baris[kol] = int(kol == f"stasiun_{st_pilih}")
    x_baru = pd.DataFrame([baris])[hasil["kolom"]]
    prob = hasil["model"].predict_proba(x_baru)[0]
    kelas_model = list(hasil["model"].classes_)
    terbaik = kelas_model[int(np.argmax(prob))]
    st.success(f"Prediksi kategori: **{terbaik}** (keyakinan {prob.max() * 100:.0f}%)")
    pr = pd.DataFrame({"kategori": kelas_model, "peluang": prob * 100})
    fig = px.bar(pr, x="peluang", y="kategori", orientation="h", color="kategori", color_discrete_map=WARNA_KAT,
                 labels={"peluang": "Peluang (%)", "kategori": ""})
    fig.update_layout(showlegend=False)
    fig.update_xaxes(range=[0, 100])
    st.plotly_chart(gaya(fig, 260), use_container_width=True)
