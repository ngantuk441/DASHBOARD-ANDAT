"""Fungsi analisis (tanpa Streamlit) supaya mudah dites dan dipakai ulang."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

POLUTAN = ["pm_sepuluh", "pm_duakomalima", "sulfur_dioksida",
           "karbon_monoksida", "ozon", "nitrogen_dioksida"]
LABEL_POLUTAN = {
    "pm_sepuluh": "PM10", "pm_duakomalima": "PM2.5", "sulfur_dioksida": "SO₂",
    "karbon_monoksida": "CO", "ozon": "O₃", "nitrogen_dioksida": "NO₂",
}
CUACA = ["suhu", "kelembapan", "angin", "hujan", "radiasi"]
LABEL_CUACA = {
    "suhu": "Suhu (°C)", "kelembapan": "Kelembapan (%)", "angin": "Kecepatan angin (m/s)",
    "hujan": "Curah hujan (mm/hari)", "radiasi": "Radiasi matahari (W/m²)",
}
URUTAN_KATEGORI = ["BAIK", "SEDANG", "TIDAK SEHAT", "SANGAT TIDAK SEHAT"]


def muat_data(folder="data"):
    ispu = pd.read_csv(f"{folder}/ispu_bersih.csv", parse_dates=["tanggal"])
    meteo = pd.read_csv(f"{folder}/meteo_harian.csv", parse_dates=["tanggal"])
    return ispu, meteo


def gabung(ispu, meteo):
    """Gabungkan ISPU dan cuaca per tanggal + stasiun (hanya periode yang sama-sama ada)."""
    return ispu.merge(meteo, on=["tanggal", "stasiun"], how="inner")


def korelasi(df, kolom_cuaca=CUACA):
    kolom = kolom_cuaca + ["ispu"]
    return df[kolom].corr(method="spearman")


def fitur_klaster(df, level):
    """level: 'Stasiun' (1 baris per stasiun) atau 'Stasiun-Bulan' (1 baris per stasiun per bulan)."""
    d = df.copy()
    d["bulan"] = d["tanggal"].dt.to_period("M").astype(str)
    kunci = ["stasiun"] if level == "Stasiun" else ["stasiun", "bulan"]
    agg = d.groupby(kunci)[POLUTAN + ["ispu"]].mean().reset_index()
    if level == "Stasiun":
        agg["bulan"] = "semua"
    return agg


def jalankan_klaster(agg, k):
    X = agg[POLUTAN]
    pipa = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Z = pipa.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    # urutkan klaster dari rata-rata ISPU terendah ke tertinggi
    urut = agg.assign(_c=km.labels_).groupby("_c")["ispu"].mean().sort_values().index
    peta = {c: i + 1 for i, c in enumerate(urut)}
    hasil = agg.copy()
    hasil["klaster"] = ["Klaster %d" % peta[c] for c in km.labels_]
    pc = PCA(n_components=2, random_state=42).fit(Z)
    koord = pc.transform(Z)
    hasil["PC1"], hasil["PC2"] = koord[:, 0], koord[:, 1]
    sil = silhouette_score(Z, km.labels_) if len(set(km.labels_)) > 1 else np.nan
    return hasil, sil, pc.explained_variance_ratio_


def tabel_silhouette(agg, k_maks=6):
    Z = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(agg[POLUTAN])
    baris = []
    for k in range(2, min(k_maks, len(agg) - 1) + 1):
        lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Z)
        baris.append({"k": k, "silhouette": silhouette_score(Z, lab)})
    return pd.DataFrame(baris)


def siapkan_klasifikasi(df, pakai_stasiun=False):
    d = df.dropna(subset=POLUTAN, how="all").copy()
    d["kategori_model"] = d["kategori"].replace({"SANGAT TIDAK SEHAT": "TIDAK SEHAT"})
    X = d[POLUTAN].copy()
    if pakai_stasiun:
        X = pd.concat([X, pd.get_dummies(d["stasiun"], prefix="stasiun").astype(int)], axis=1)
    return X, d["kategori_model"], d


def latih_klasifikasi(df, ukuran_uji=0.25, pakai_stasiun=False, n_pohon=200):
    X, y, d = siapkan_klasifikasi(df, pakai_stasiun)
    strat = y if y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=ukuran_uji, random_state=42, stratify=strat
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=n_pohon, random_state=42, class_weight="balanced"),
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    rf = model.named_steps["randomforestclassifier"]
    kelas = [k for k in URUTAN_KATEGORI if k in set(rf.classes_)]
    cm = confusion_matrix(yte, pred, labels=kelas)
    penting = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    per_stasiun = (
        pd.DataFrame({"stasiun": d.loc[yte.index, "stasiun"], "benar": pred == yte.values})
        .groupby("stasiun")["benar"].agg(["mean", "size"]).reset_index()
        .rename(columns={"mean": "akurasi", "size": "n_uji"})
    )
    return {
        "model": model, "kelas": kelas, "cm": cm, "penting": penting,
        "akurasi": accuracy_score(yte, pred),
        "f1_makro": f1_score(yte, pred, average="macro"),
        "n_latih": len(Xtr), "n_uji": len(Xte), "per_stasiun": per_stasiun,
        "kolom": list(X.columns),
    }
