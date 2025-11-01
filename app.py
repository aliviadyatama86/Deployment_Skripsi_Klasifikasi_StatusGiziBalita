# app.py
# ============================================================
# Aplikasi Streamlit:
# Klasifikasi Status Gizi Balita (Random Forest)
# Fitur:
# 1) Analisis Perkembangan Balita per Tahun (upload Excel + pilih nama + tahun)
# 2) Prediksi Manual (form input nilai Z-Score + status dropdown)
# 3) Histori pengukuran & Reset form
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO
from datetime import datetime

# -----------------------------
# Config page
# -----------------------------
st.set_page_config(page_title="Prediksi Status Gizi Balita", page_icon="🧒", layout="wide")
st.title("🧒 Prediksi & Pemantauan Status Gizi Balita")

st.markdown("""
**Fitur aplikasi**
- Upload file Excel berisi pengukuran balita.
- Analisis perkembangan per tahun (ambil pengukuran terakhir tiap bulan).
- Prediksi status gizi manual (isi form) menggunakan model Random Forest (jika tersedia).
- Simpan & lihat histori hasil prediksi.
""")

# -----------------------------
# Load model (jika ada)
# -----------------------------
MODEL_FILE = "final_model_status_gizi.sav"
model = None
model_features = None
if os.path.exists(MODEL_FILE):
    try:
        bundle = joblib.load(MODEL_FILE)
        model = bundle.get("model", None)
        model_features = bundle.get("features", None)
        st.success("✅ Model ditemukan dan berhasil dimuat.")
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat model: {e}")
else:
    st.info("⚠️ Model tidak ditemukan. Prediksi otomatis menggunakan model akan dinonaktifkan sampai file model diunggah.")

# -----------------------------
# Utility functions
# -----------------------------
def clean_and_prepare_df(df):
    """Standarisasi nama kolom & tipe untuk dataframe yang diupload."""
    df = df.copy()
    # Pastikan kolom yang dipakai ada; jika belum, coba menormalisasi nama kolom umum
    df.columns = [c.strip() for c in df.columns]
    # Parse tanggal
    if "Tanggal Pengukuran" in df.columns:
        df["Tanggal Pengukuran"] = pd.to_datetime(df["Tanggal Pengukuran"], errors="coerce", dayfirst=True)
    # Replace comma decimal
    for col in ["Z-Score BB/U", "Z-Score TB/U", "Z-Score BB/TB", "BB", "TB"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".")
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Normalisasi nama anak stripping
    if "Nama Anak" in df.columns:
        df["Nama Anak"] = df["Nama Anak"].astype(str).str.strip()
    return df

def last_measurement_per_month(df):
    """Ambil pengukuran terakhir tiap bulan per anak (periode bulanan)."""
    df = df.copy()
    if "Tanggal Pengukuran" not in df.columns:
        return pd.DataFrame()
    df["Periode"] = df["Tanggal Pengukuran"].dt.to_period("M")
    df = df.sort_values(["Nama Anak", "Tanggal Pengukuran"])
    df_last = df.groupby(["Nama Anak", "Periode"], as_index=False).last()
    # convert Periode ke timestamp (ambil start of month)
    df_last["Periode_MonthStart"] = df_last["Periode"].dt.to_timestamp()
    return df_last

def interpret_trend(z_series):
    """Interpretasi sederhana trend: stable/good, naik turun, buruk (per bulan).
       z_series: pd.Series indexed by month (chronological).
    """
    if z_series.empty:
        return "Tidak ada data untuk tahun ini."
    # categorize each zscore into band
    def category(z):
        if pd.isna(z):
            return np.nan
        if z < -3:
            return "Sangat Buruk"
        if -3 <= z < -2:
            return "Gizi Buruk"
        if -2 <= z < -1:
            return "Gizi Kurang"
        if -1 <= z < 2:
            return "Gizi Baik"
        if 2 <= z < 3:
            return "Risiko Gizi Lebih"
        if z >= 3:
            return "Gizi Lebih / Obesitas"
        return "Unknown"

    cats = z_series.map(category).dropna().tolist()
    if not cats:
        return "Data Z-Score tidak mencukupi untuk analisis."

    # If any month in Gizi Buruk or Sangat Buruk -> urgent
    if any(c in ["Sangat Buruk", "Gizi Buruk"] for c in cats):
        return "Perhatian: Ada periode dengan status Gizi Buruk → perlu tindakan/monitoring segera."

    # Count distinct categories
    distinct = set(cats)
    # If mostly 'Gizi Baik' and few variations -> stable baik
    prop_baik = cats.count("Gizi Baik") / len(cats)
    # If many switches between categories -> naik/turun
    changes = sum(1 for i in range(1, len(cats)) if cats[i] != cats[i-1])

    if prop_baik >= 0.75 and changes <= 1:
        return "Gizi baik dan relatif stabil sepanjang tahun."
    if changes >= 3:
        return "Naik-turun cukup sering — perlu pemantauan berkala oleh tenaga kesehatan."
    # Else
    return "Cenderung fluktuatif — disarankan pemantauan dan intervensi jika perlu."

def ensure_history_file():
    if "riwayat_prediksi.csv" not in os.listdir():
        pd.DataFrame([], columns=["Nama","Tanggal","Prediksi"] + (model_features or [])).to_csv("riwayat_prediksi.csv", index=False)

# -----------------------------
# UI: Tabs for 3 features
# -----------------------------
tab1, tab2, tab3 = st.tabs(["1. Perkembangan per Tahun", "2. Prediksi Manual", "3. Histori & Reset"])

# -----------------------------
# TAB 1: Perkembangan per Tahun
# -----------------------------
with tab1:
    st.header("🔎 Analisis Perkembangan Balita per Tahun")
    uploaded_file = st.file_uploader("Upload file Excel (contoh .xlsx) — gunakan kolom sesuai template", type=["xlsx", "xls"], key="upload_tab1")
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            df = clean_and_prepare_df(df)
            st.success("File berhasil diunggah dan dibaca.")
            st.write("Preview data (5 baris):")
            st.dataframe(df.head())

            # Pilih nama & tahun
            if "Nama Anak" not in df.columns or "Tanggal Pengukuran" not in df.columns:
                st.error("Kolom 'Nama Anak' dan/atau 'Tanggal Pengukuran' tidak ditemukan. Pastikan nama kolom sesuai.")
            else:
                names = df["Nama Anak"].dropna().unique().tolist()
                nama_pilih = st.selectbox("Pilih Nama Anak", options=names)
                tahun_pilih = st.number_input("Masukkan Tahun (contoh: 2021)", min_value=2000, max_value=2100, value=datetime.now().year)
                # compute last per month
                df_last = last_measurement_per_month(df)
                # filter by name and year
                df_person = df_last[df_last["Nama Anak"].str.lower() == str(nama_pilih).lower()]
                df_person_year = df_person[df_person["Periode_MonthStart"].dt.year == int(tahun_pilih)]
                if df_person_year.empty:
                    st.info("Tidak ditemukan pengukuran untuk nama & tahun yang dipilih.")
                else:
                    # reindex months 1..12 for display
                    df_person_year = df_person_year.sort_values("Periode_MonthStart")
                    display_cols = ["Periode_MonthStart", "Tanggal Pengukuran", "BB", "TB", "Z-Score BB/TB", "Status BB/TB"]
                    st.subheader(f"📋 Pengukuran terakhir tiap bulan — {nama_pilih} ({tahun_pilih})")
                    st.dataframe(df_person_year[display_cols].reset_index(drop=True))

                    # Plot Z-Score BB/TB trend
                    st.subheader("📈 Grafik Perkembangan Z-Score BB/TB (per bulan)")
                    series = df_person_year.set_index("Periode_MonthStart")["Z-Score BB/TB"].dropna()
                    # ensure monthly index (for months without data, we'll show gaps)
                    all_months = pd.period_range(start=f"{tahun_pilih}-01", end=f"{tahun_pilih}-12", freq="M").to_timestamp()
                    ser_full = pd.Series(index=all_months, dtype=float)
                    for idx, val in series.items():
                        ser_full.loc[idx] = val
                    st.line_chart(ser_full)

                    # Interpretation
                    interp = interpret_trend(series)
                    st.markdown("**Interpretasi singkat:**")
                    st.info(interp)

                    # Allow download of filtered data
                    to_download = BytesIO()
                    df_person_year.to_excel(to_download, index=False, sheet_name="Perkembangan")
                    st.download_button("💾 Unduh data perkembangan (Excel)", to_download.getvalue(),
                                       file_name=f"perkembangan_{nama_pilih.replace(' ','_')}_{tahun_pilih}.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
    else:
        st.info("Unggah file Excel berisi data pengukuran terlebih dahulu.")

# -----------------------------
# TAB 2: Prediksi Manual (Form)
# -----------------------------
with tab2:
    st.header("🧾 Prediksi Manual — Masukkan Pengukuran Terakhir")
    st.markdown("Isi form berikut untuk memprediksi status gizi balita (model Random Forest akan digunakan jika tersedia).")
    # Form inputs
    with st.form("form_prediksi", clear_on_submit=False):
        nama_input = st.text_input("Nama Balita")
        tanggal_input = st.date_input("Tanggal Pengukuran", value=datetime.today())
        z_bbtb = st.text_input("Z-Score BB/TB (contoh: 0.5)", key="z_bbtb")
        z_bbu = st.text_input("Z-Score BB/U (contoh: -0.8)", key="z_bbu")
        z_tbu = st.text_input("Z-Score TB/U (contoh: 0.2)", key="z_tbu")

        # Dropdown for status BB/U
        opsi_bbu = ["Sangat Kurang", "Kurang", "Normal", "Risiko Gizi Lebih"]
        opsi_tbu = ["Sangat Pendek", "Pendek", "Normal", "Tinggi"]
        status_bbu = st.selectbox("Status BB/U (pilih yang sesuai)", options=opsi_bbu)
        status_tbu = st.selectbox("Status TB/U (pilih yang sesuai)", options=opsi_tbu)

        submit_pred = st.form_submit_button("🔍 Prediksi")

    if submit_pred:
        # basic validation & convert
        try:
            z_bbtb_val = float(str(z_bbtb).replace(",", "."))
            z_bbu_val = float(str(z_bbu).replace(",", "."))
            z_tbu_val = float(str(z_tbu).replace(",", "."))
        except:
            st.error("Pastikan Z-Score diisi dengan format angka (gunakan titik '.' untuk desimal).")
            z_bbtb_val = z_bbu_val = z_tbu_val = None

        # mapping statuses to encoded (sesuai yang dipakai saat training)
        mapping_status_bbu = {
            'Sangat Kurang': 0,
            'Kurang': 1,
            'Normal': 2,
            'Risiko Gizi Lebih': 3
        }
        mapping_status_tbu = {
            'Sangat Pendek': 0,
            'Pendek': 1,
            'Normal': 2,
            'Tinggi': 3
        }

        if None in (z_bbtb_val, z_bbu_val, z_tbu_val):
            st.warning("Perbaiki input z-score terlebih dahulu.")
        else:
            # build input df for model. Use model_features if present, else use default feature order:
            feature_order = model_features or [
                'Z-Score BB/TB', 'Z-Score BB/U', 'Status BB/U (Encoded)', 'Status TB/U (Encoded)', 'Z-Score TB/U'
            ]
            # prepare a dict with expected keys
            row = {}
            for f in feature_order:
                if f == 'Z-Score BB/TB':
                    row[f] = z_bbtb_val
                elif f == 'Z-Score BB/U':
                    row[f] = z_bbu_val
                elif f == 'Z-Score TB/U':
                    row[f] = z_tbu_val
                elif f in ['Status BB/U (Encoded)', 'Status BB/U']:
                    row[f] = mapping_status_bbu.get(status_bbu, 2)
                elif f in ['Status TB/U (Encoded)', 'Status TB/U']:
                    row[f] = mapping_status_tbu.get(status_tbu, 2)
                else:
                    # if unexpected feature appears, try to set 0
                    row[f] = 0

            X_input = pd.DataFrame([row], columns=feature_order)

            if model is None:
                st.warning("Model tidak tersedia — prediksi otomatis tidak dapat dilakukan. Namun data input akan disimpan ke histori.")
                predicted_label = None
                probabilities = None
            else:
                try:
                    predicted = model.predict(X_input)[0]
                    probs = model.predict_proba(X_input)[0]
                    predicted_label = int(predicted)
                    probabilities = probs
                except Exception as e:
                    st.error(f"Gagal melakukan prediksi menggunakan model: {e}")
                    predicted_label = None
                    probabilities = None

            # map predicted label to text (complete mapping)
            label_map = {
                0: "Gizi Buruk",
                1: "Gizi Kurang",
                2: "Gizi Baik",
                3: "Risiko Gizi Lebih",
                4: "Gizi Lebih",
                5: "Obesitas"
            }
            hasil_text = label_map.get(predicted_label, "Tidak tersedia")

            st.subheader("🔎 Hasil Prediksi")
            if predicted_label is not None and probabilities is not None:
                st.success(f"Hasil: {hasil_text}  — Prob: {probabilities[predicted_label]:.2%}")
                # show full probability breakdown if model's classes match
                classes = getattr(model, "classes_", None)
                if classes is not None and len(classes) == len(probabilities):
                    # make readable
                    prob_map = {label_map.get(int(c), str(c)): f"{probabilities[i]:.2%}" for i, c in enumerate(classes)}
                    st.table(pd.DataFrame.from_dict(prob_map, orient="index", columns=["Probabilitas"]))
            else:
                st.info("Hasil prediksi model tidak tersedia.")

            # Save to history CSV & session
            rec = {
                "Nama": nama_input or "",
                "Tanggal": pd.to_datetime(tanggal_input).strftime("%Y-%m-%d"),
                "Z-Score BB/TB": z_bbtb_val,
                "Z-Score BB/U": z_bbu_val,
                "Z-Score TB/U": z_tbu_val,
                "Status BB/U (Encoded)": mapping_status_bbu.get(status_bbu, 2),
                "Status TB/U (Encoded)": mapping_status_tbu.get(status_tbu, 2),
                "Prediksi_Label": hasil_text
            }
            # append to CSV
            if not os.path.exists("riwayat_prediksi.csv"):
                pd.DataFrame([rec]).to_csv("riwayat_prediksi.csv", index=False)
            else:
                pd.DataFrame([rec]).to_csv("riwayat_prediksi.csv", mode="a", header=False, index=False)
            st.success("✅ Hasil prediksi disimpan ke riwayat (riwayat_prediksi.csv)")

# -----------------------------
# TAB 3: Histori Pengukuran & Reset
# -----------------------------
with tab3:
    st.header("📚 Histori Prediksi & Reset Form")
    if os.path.exists("riwayat_prediksi.csv"):
        df_hist = pd.read_csv("riwayat_prediksi.csv")
        st.subheader("Riwayat Prediksi (Terakhir)")
        st.dataframe(df_hist.sort_values("Tanggal", ascending=False).reset_index(drop=True))
        # Download history
        todl = BytesIO()
        df_hist.to_excel(todl, index=False, sheet_name="Riwayat")
        st.download_button("💾 Unduh Riwayat (Excel)", todl.getvalue(), file_name="riwayat_prediksi.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if st.button("🔄 Reset / Hapus Riwayat (riwayat_prediksi.csv)"):
            os.remove("riwayat_prediksi.csv")
            st.success("Riwayat berhasil dihapus.")
    else:
        st.info("Belum ada riwayat. Lakukan prediksi lewat tab Prediksi Manual atau unggah file dan lakukan prediksi.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Catatan: Aplikasi ini bersifat alat bantu. Interpretasi dan intervensi medis harus dikonsultasikan dengan tenaga kesehatan.")
