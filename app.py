import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

# === Load model ===
bundle = joblib.load("final_klasifikasi_gizi_balita.sav")
model = bundle["model"]
selected_features = bundle["features"]

# === Judul Aplikasi ===
st.title("🧒 Prediksi Status Gizi Balita Menggunakan Random Forest")
st.markdown("Masukkan data hasil **pengukuran posyandu terakhir** untuk memprediksi status gizi balita.")
st.caption("Sistem ini membantu memantau perkembangan gizi balita berdasarkan nilai Z-Score setiap bulan.")

# === Inisialisasi session state ===
if "history" not in st.session_state:
    st.session_state.history = []

# === Form Input ===
st.subheader("📋 Form Input Data Balita")

nama = st.text_input("Nama Balita")
tanggal = st.date_input("Tanggal Pengukuran", datetime.date.today())

user_input = {}
for feature in selected_features:
    val = st.text_input(f"{feature}", "")
    if val.strip() == "":
        user_input[feature] = None
    else:
        try:
            val = val.replace(",", ".")
            user_input[feature] = float(val)
        except ValueError:
            st.error(f"Input {feature} harus berupa angka!")
            user_input[feature] = None

# === Tombol aksi ===
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    pred_btn = st.button("🔍 Prediksi Status Gizi")
with col2:
    reset_btn = st.button("🔁 Reset Form")
with col3:
    grafik_btn = st.button("📈 Lihat Grafik Perkembangan")

# === Reset Form ===
if reset_btn:
    for feature in selected_features:
        if feature in st.session_state:
            del st.session_state[feature]
    st.rerun()

# === Prediksi ===
if pred_btn:
    if not nama:
        st.warning("⚠️ Harap isi nama balita terlebih dahulu.")
    elif any(v is None for v in user_input.values()):
        st.warning("⚠️ Harap isi semua data numerik.")
    else:
        input_df = pd.DataFrame([user_input], columns=selected_features)

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # === Mapping hasil prediksi lengkap ===
        status_dict = {
            0: "Gizi Buruk",
            1: "Gizi Kurang",
            2: "Gizi Baik",
            3: "Risiko Gizi Lebih",
            4: "Gizi Lebih",
            5: "Obesitas"
        }
        hasil_status = status_dict.get(prediction, "Tidak Diketahui")

        # === Tampilan hasil ===
        st.markdown("---")
        st.markdown(
            f"<h2 style='text-align:center; color:#4CAF50;'>📊 Hasil Prediksi: {hasil_status}</h2>"
            f"<h4 style='text-align:center;'>Probabilitas: {probabilities[prediction]:.2%}</h4>",
            unsafe_allow_html=True
        )

        # === Rekomendasi berdasarkan status ===
        rekomendasi = {
            "Gizi Baik": "Pertahankan pola makan seimbang dan rutin datang ke posyandu.",
            "Gizi Kurang": "Perbanyak asupan protein dan kalori, serta pantau pertumbuhan secara rutin.",
            "Gizi Buruk": "Segera konsultasikan ke petugas gizi atau puskesmas terdekat.",
            "Risiko Gizi Lebih": "Perhatikan asupan makanan, kurangi makanan tinggi gula dan lemak.",
            "Gizi Lebih": "Batasi makanan berlemak dan tinggi kalori, serta ajak anak beraktivitas fisik.",
            "Obesitas": "Konsultasikan dengan petugas gizi untuk penanganan lebih lanjut."
        }

        st.info(f"🧾 *Rekomendasi:* {rekomendasi.get(hasil_status, 'Tidak ada rekomendasi khusus.')}")

        # === Simpan ke riwayat CSV ===
        record = {"Nama": nama, "Tanggal": tanggal, "Prediksi": hasil_status, **user_input}
        st.session_state.history.append(record)

        if not os.path.exists("riwayat_prediksi.csv"):
            pd.DataFrame(st.session_state.history).to_csv("riwayat_prediksi.csv", index=False)
        else:
            pd.DataFrame([record]).to_csv("riwayat_prediksi.csv", mode='a', header=False, index=False)

        # === Visualisasi probabilitas ===
        st.subheader("📈 Distribusi Probabilitas Prediksi")
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#e74c3c', '#f39c12', '#2ecc71', '#f1c40f', '#3498db', '#9b59b6']
        ax.bar(status_dict.values(), probabilities, color=colors)
        ax.set_ylabel("Probabilitas")
        ax.set_ylim(0, 1)
        ax.set_xticklabels(status_dict.values(), rotation=30, ha='right')
        for i, v in enumerate(probabilities):
            ax.text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=10)
        st.pyplot(fig)

# === Grafik Perkembangan Individu ===
if grafik_btn:
    if not nama:
        st.warning("⚠️ Masukkan nama balita terlebih dahulu.")
    elif os.path.exists("riwayat_prediksi.csv"):
        df = pd.read_csv("riwayat_prediksi.csv")
        df_nama = df[df["Nama"].str.lower() == nama.lower()]

        if len(df_nama) > 0:
            st.subheader(f"📊 Grafik Perkembangan Z-Score — {nama}")

            df_nama["Tanggal"] = pd.to_datetime(df_nama["Tanggal"])
            df_nama = df_nama.sort_values("Tanggal")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df_nama["Tanggal"], df_nama["Z-Score BB/TB"], marker='o', label="BB/TB")
            ax.plot(df_nama["Tanggal"], df_nama["Z-Score BB/U"], marker='s', label="BB/U")
            ax.plot(df_nama["Tanggal"], df_nama["Z-Score TB/U"], marker='^', label="TB/U")
            
            # Warna zona status gizi (berdasarkan BB/TB)
            ax.axhspan(-3, -2, color='#ffcccc', alpha=0.4, label="Gizi Buruk")
            ax.axhspan(-2, -1, color='#ffe0b2', alpha=0.4, label="Gizi Kurang")
            ax.axhspan(-1, 1, color='#dcedc8', alpha=0.4, label="Gizi Baik")
            ax.axhspan(1, 2, color='#fff59d', alpha=0.4, label="Risiko Gizi Lebih")
            ax.axhspan(2, 3, color='#bbdefb', alpha=0.4, label="Gizi Lebih")
            ax.axhspan(3, 5, color='#d1c4e9', alpha=0.4, label="Obesitas")

            ax.set_xlabel("Tanggal Pengukuran")
            ax.set_ylabel("Nilai Z-Score")
            ax.set_title(f"Perkembangan Z-Score Balita ({nama})")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            st.caption("📘 Garis warna menunjukkan rentang kategori gizi menurut Permenkes No. 2 Tahun 2020.")
        else:
            st.info(f"Belum ada data prediksi untuk balita bernama {nama}.")
    else:
        st.info("Belum ada data riwayat tersimpan.")


