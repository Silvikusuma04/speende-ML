from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Memuat model, encoder, dan scaler
model = load_model('model_loan.h5')

with open('mapping\scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)

with open('mapping\label_encoder.pkl', 'rb') as f:
    le = joblib.load(f)

with open('mapping\ordinal_encoder.pkl', 'rb') as f:
    oe = joblib.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Ambil input dari form HTML sesuai dengan urutan kolom yang benar
        total_debt_income_ratio = float(request.form['Total_Utang_Terhadap_Pendapatan'])
        monthly_income = float(request.form['Pendapatan_Bulanan'])
        annual_income = float(request.form['Pendapatan_Tahunan'])
        interest_rate = float(request.form['Suku_Bunga_Yang_Diterapkan'])
        loan_amount = float(request.form['Jumlah_Pinjaman'])
        base_interest_rate = float(request.form['Suku_Bunga_Awal'])
        education_level = request.form['Tingkat_Pendidikan']
        net_worth = float(request.form['Kekayaan_Bersih'])
        monthly_loan_payment = float(request.form['Pembayaran_Pinjaman_Bulanan'])
        total_assets = float(request.form['Total_Aset'])
        age = int(request.form['Usia_Pemohon'])
        credit_score = int(request.form['Skor_Kelayakan_Kredit'])
        experience = int(request.form['Pengalaman_Kerja'])
        credit_history_duration = int(request.form['Durasi_Sejarah_Kredit'])
        loan_payment_period = int(request.form['Periode_Pembayaran_Pinjaman'])
        monthly_debt_payments = float(request.form['Pembayaran_Utang_Bulanan'])
        savings_balance = float(request.form['Saldo_Tabungan'])
        credit_inquiries = int(request.form['Jumlah_Pengecekan_Kredit'])
        dependents = int(request.form['Jumlah_Tanggungan'])
        active_credit_lines = int(request.form['Jumlah_Jalur_Kredit_Aktif'])


        # Membuat DataFrame dari input, urutan kolom sesuai dengan gambar yang kamu kirim
        column_order = [
            'Total_Utang_Terhadap_Pendapatan', 'Pendapatan_Bulanan', 'Pendapatan_Tahunan', 'Suku_Bunga_Yang_Diterapkan',
            'Jumlah_Pinjaman', 'Suku_Bunga_Awal', 'Tingkat_Pendidikan', 'Kekayaan_Bersih', 'Pembayaran_Pinjaman_Bulanan', 
            'Total_Aset', 'Usia_Pemohon', 'Skor_Kelayakan_Kredit', 'Pengalaman_Kerja', 'Durasi_Sejarah_Kredit',
            'Periode_Pembayaran_Pinjaman', 'Pembayaran_Utang_Bulanan','Saldo_Tabungan', 'Jumlah_Pengecekan_Kredit', 'Jumlah_Tanggungan', 'Jumlah_Jalur_Kredit_Aktif'
        ]
        
        new_data = pd.DataFrame([{
            'Total_Utang_Terhadap_Pendapatan': total_debt_income_ratio,
            'Pendapatan_Bulanan': monthly_income,
            'Pendapatan_Tahunan': annual_income,
            'Suku_Bunga_Yang_Diterapkan': interest_rate,
            'Jumlah_Pinjaman': loan_amount,
            'Suku_Bunga_Awal': base_interest_rate,
            'Tingkat_Pendidikan': education_level,
            'Kekayaan_Bersih': net_worth,
            'Pembayaran_Pinjaman_Bulanan': monthly_loan_payment,
            'Total_Aset': total_assets,
            'Usia_Pemohon': age,
            'Skor_Kelayakan_Kredit': credit_score,
            'Pengalaman_Kerja': experience,
            'Durasi_Sejarah_Kredit': credit_history_duration,
            'Periode_Pembayaran_Pinjaman': loan_payment_period,
            'Pembayaran_Utang_Bulanan': monthly_debt_payments,
            'Saldo_Tabungan': savings_balance,
            'Jumlah_Pengecekan_Kredit': credit_inquiries,
            'Jumlah_Tanggungan': dependents,
            'Jumlah_Jalur_Kredit_Aktif': active_credit_lines
        }])

        # Menyusun kolom sesuai urutan yang digunakan pada pelatihan
        new_data = new_data[column_order]

        # Encoding data kategorikal
        new_data['Tingkat_Pendidikan'] = oe.transform(new_data[['Tingkat_Pendidikan']])


        # Standarisasi data
        new_data_scaled = scaler.transform(new_data)

        # Prediksi
        prediksi_proba = model.predict(new_data_scaled)
        prediksi_label = (prediksi_proba > 0.5).astype(int)

        # Menampilkan hasil prediksi
        result = 'Disetujui' if prediksi_label[0][0] == 1 else 'Ditolak'
        prob = prediksi_proba[0][0]

        return render_template('index.html', prediction_text=f"Prediksi Status Pinjaman: {result} (Probabilitas: {prob:.4f})")

if __name__ == "__main__":
    app.run(debug=True)


if __name__ == "__main__":
    app.run(debug=True)
