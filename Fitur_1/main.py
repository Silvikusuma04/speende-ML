from flask import Flask, request, render_template_string, send_from_directory
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import shap
import random
import os

app = Flask(__name__)

# Load model dan preprocessing
model = load_model('model/best_model.h5')
le = joblib.load('Mapping/label_encoder_kategori.pkl')
scaler = joblib.load('Mapping/scaler_startup_success.pkl')

# SHAP background
background = np.zeros((10, len(scaler.feature_names_in_) + 1))
def model_predict(X):
    return model.predict(X).flatten()
explainer = shap.KernelExplainer(model_predict, background)

# Load HTML template dari file
with open("index.html", "r") as f:
    html_template = f.read()

@app.route('/style.css')
def serve_css():
    return send_from_directory('.', 'style.css')

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

    def generate_reason_with_values(shap_vals, features, label_output, original_values, n_random=5):
        shap_list = list(zip(features, shap_vals[0]))
        sampled = random.sample(shap_list, min(n_random, len(shap_list)))
        reasons = []
        for feat, val in sampled:
            arah = "meningkatkan" if val > 0 else "menurunkan"
            nilai_asli = original_values.get(feat, 0)
            if isinstance(nilai_asli, str):
                nilai_fmt = nilai_asli
            elif abs(nilai_asli) >= 1000:
                nilai_fmt = f"{int(nilai_asli):,}"
            else:
                nilai_fmt = f"{nilai_asli:.2f}"
            reasons.append(f"{feat.replace('_', ' ')} ({nilai_fmt}) {arah} kemungkinan {label_output.lower()}")
        return " dan ".join(reasons)

    reason = generate_reason_with_values(shap_array, list(scaler.feature_names_in_) + ['populer'], label, inverse_values)
    return label, reason

@app.route("/", methods=["GET", "POST"])
def index():
    result, reason = None, None
    if request.method == "POST":
        data_input = {
            'umur_milestone_terakhir': float(request.form['umur_milestone_terakhir']),
            'relasi': int(request.form['relasi']),
            'umur_pendanaan_pertama': float(request.form['umur_pendanaan_pertama']),
            'total_dana': float(request.form['total_dana']),
            'umur_pendanaan_terakhir': float(request.form['umur_pendanaan_terakhir']),
            'umur_milestone_pertama': float(request.form['umur_milestone_pertama']),
            'rata_partisipan': int(request.form['rata_partisipan']),
            'kategori': request.form['kategori'],
            'jumlah_pendanaan': int(request.form['jumlah_pendanaan']),
            'jumlah_milestone': int(request.form['jumlah_milestone']),
            'rasio_dana_per_relasi': float(request.form['rasio_dana_per_relasi']),
            'dana_per_pendanaan': float(request.form['dana_per_pendanaan']),
            'populer': int(request.form['populer'])
        }
        result, reason = predict_and_explain(data_input)
    return render_template_string(html_template, result=result, reason=reason)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
