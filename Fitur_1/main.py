from flask import Flask, request, render_template_string, send_from_directory
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import shap
import random
import os
from datetime import datetime
from flask_cors import CORS
from flask import jsonify

app = Flask(__name__)

CORS(app)

model = load_model('model/best_model.h5')
le = joblib.load('Mapping/label_encoder_kategori.pkl')
scaler = joblib.load('Mapping/scaler_startup_success.pkl')

background = np.zeros((10, len(scaler.feature_names_in_) + 1))
def model_predict(X):
    return model.predict(X).flatten()
explainer = shap.KernelExplainer(model_predict, background)

with open("index.html", "r") as f:
    html_template = f.read()

@app.route('/style.css')
def serve_css():
    return send_from_directory('.', 'style.css')

def calculate_age_in_years(date_str):
    today = datetime.today()
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        delta = today - date_obj
        return round(delta.days / 365, 2)
    except ValueError:
        return 0.0

def generate_reason_with_values(shap_vals, features, label_output, original_values, n_random=5):
    display_labels = {
        'umur_milestone_terakhir': 'Umur Pencapaian Terakhir',
        'relasi': 'Jumlah Relasi atau Investor',
        'umur_pendanaan_pertama': 'Umur Pendanaan Pertama',
        'total_dana': 'Total Dana yang Dimiliki',
        'umur_pendanaan_terakhir': 'Umur Pendanaan Terakhir',
        'umur_milestone_pertama': 'Umur Pencapaian Awal',
        'rata_partisipan': 'Rata-rata Partisipan atau Pelanggan',
        'jumlah_pendanaan': 'Jumlah Penerimaan Pendanaan',
        'jumlah_milestone': 'Jumlah Pencapaian',
        'rasio_dana_per_relasi': 'Rasio Dana diinvestasikan per Relasi',
        'dana_per_pendanaan': 'Rata-rata Dana Tiap Pendanaan',
        'populer': 'Status Populer'
    }
    positive_reasons = []
    negative_reasons = []
    shap_list = list(zip(features, shap_vals[0]))
    shap_list_sorted = sorted(shap_list, key=lambda x: abs(x[1]), reverse=True)

    for feat, val in shap_list_sorted:
        arah = "mendukung" if val > 0 else "mengurangi"
        nilai_asli = original_values.get(feat, 0)
        satuan = " tahun" if "umur" in feat else ""
        if isinstance(nilai_asli, str):
            nilai_fmt = nilai_asli
        elif abs(nilai_asli) >= 1_000_000:
            nilai_fmt = f"{int(nilai_asli / 1_000_000):,} juta"
        elif abs(nilai_asli) >= 1_000:
            nilai_fmt = f"{int(nilai_asli):,}"
        else:
            nilai_fmt = f"{nilai_asli:.2f}"
        label_feat = display_labels.get(feat, feat.replace('_', ' '))
        if label_output.lower() == 'gagal':
            arah_text = 'mengurangi risiko kegagalan' if val > 0 else 'meningkatkan risiko kegagalan'
        else:
            arah_text = 'mendukung potensi sukses' if val > 0 else 'mengurangi potensi sukses'
        reason_text = f"'{label_feat}' ({nilai_fmt}{satuan}) {arah_text}"
        if val > 0:
            positive_reasons.append(reason_text)
        else:
            negative_reasons.append(reason_text)
        

    return positive_reasons, negative_reasons

def predict_and_explain(data_dict):
    df = pd.DataFrame([data_dict])
    df['kategori_encoded'] = le.transform(df['kategori'])
    original_input = df.copy()
    df['kategori'] = df['kategori_encoded']
    df.drop(columns=['kategori_encoded'], inplace=True)

    scaled_features = scaler.transform(df[scaler.feature_names_in_])
    sample_scaled = np.concatenate([scaled_features, df[['populer']].values], axis=1)

    prediction = model.predict(sample_scaled)
    binary_result = int(prediction[0][0] > 0.5)
    label = "Sukses" if binary_result == 1 else "Gagal"

    shap_values = explainer.shap_values(sample_scaled)
    shap_array = shap_values[0] if isinstance(shap_values, list) else shap_values

    scaled_part = sample_scaled[0][:-1].reshape(1, -1)
    inversed_scaled = scaler.inverse_transform(scaled_part).flatten()
    inverse_values = dict(zip(scaler.feature_names_in_, inversed_scaled))
    inverse_values['populer'] = int(df['populer'].values[0])
    inverse_values['kategori'] = original_input['kategori'].values[0]

    pos_reason, neg_reason = generate_reason_with_values(shap_array, list(scaler.feature_names_in_) + ['populer'], label, inverse_values)
    return label, pos_reason, neg_reason

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result, pos_reason, neg_reason = None, None, None

    if request.method == "POST":
        if request.is_json:
            data = request.get_json()
            try:
                data_input = {
                    'umur_milestone_terakhir': float(data['umur_milestone_terakhir']),
                    'relasi': int(data['relasi']),
                    'umur_pendanaan_pertama': float(data['umur_pendanaan_pertama']),
                    'total_dana': float(data['total_dana']),
                    'umur_pendanaan_terakhir': float(data['umur_pendanaan_terakhir']),
                    'umur_milestone_pertama': float(data['umur_milestone_pertama']),
                    'rata_partisipan': int(data['rata_partisipan']),
                    'kategori': data['kategori'],
                    'jumlah_pendanaan': int(data['jumlah_pendanaan']),
                    'jumlah_milestone': int(data['jumlah_milestone']),
                    'rasio_dana_per_relasi': float(data['rasio_dana_per_relasi']),
                    'dana_per_pendanaan': float(data['dana_per_pendanaan']),
                    'populer': int(data['populer'])
                }
            except Exception as e:
                return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

        else:
            try:
                data_input = {
                    'umur_milestone_terakhir': calculate_age_in_years(request.form['tanggal_pencapaian_terakhir']),
                    'relasi': int(request.form['relasi']),
                    'umur_pendanaan_pertama': calculate_age_in_years(request.form['tanggal_pendanaan_pertama']),
                    'total_dana': float(request.form['total_dana']),
                    'umur_pendanaan_terakhir': calculate_age_in_years(request.form['tanggal_pendanaan_terakhir']),
                    'umur_milestone_pertama': calculate_age_in_years(request.form['tanggal_pencapaian_awal']),
                    'rata_partisipan': int(request.form['rata_partisipan']),
                    'kategori': request.form['kategori'],
                    'jumlah_pendanaan': int(request.form['jumlah_pendanaan']),
                    'jumlah_milestone': int(request.form['jumlah_capaian']),
                    'rasio_dana_per_relasi': float(request.form['rasio_dana_per_relasi']),
                    'dana_per_pendanaan': float(request.form['dana_per_pendanaan']),
                    'populer': int(request.form['populer'])
                }
            except Exception as e:
                return f"Form input error: {str(e)}", 400

        result, pos_reason, neg_reason = predict_and_explain(data_input)

        if request.is_json:
            return jsonify({
                "result": result,
                "positive_reasons": pos_reason,
                "negative_reasons": neg_reason
            })

    return render_template_string(html_template, result=result, pos_reason=pos_reason, neg_reason=neg_reason)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
