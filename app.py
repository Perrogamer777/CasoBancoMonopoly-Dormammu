from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Diccionario de regiones de Chile
REGIONES_CHILE = {
    'Arica y Parinacota': 1,
    'Tarapacá': 2,
    'Antofagasta': 3,
    'Atacama': 4,
    'Coquimbo': 5,
    'Valparaíso': 6,
    'Metropolitana': 7,
    'O\'Higgins': 8,
    'Maule': 9,
    'Ñuble': 10,
    'Biobío': 11,
    'La Araucanía': 12,
    'Los Ríos': 13,
    'Los Lagos': 14,
    'Aysén': 15,
    'Magallanes': 16
}

# Cargar modelo y scaler
modelo = joblib.load('modelo_internauta_logreg.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html', regiones=REGIONES_CHILE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del request
        data = request.get_json()
        
        # Convertir nombre de región a número
        region_num = REGIONES_CHILE[data['region']]
        
        # Crear array con características
        features = np.array([
            float(data['edad']),
            float(data['renta']),
            float(region_num),
            float(data['antiguedad'])
        ]).reshape(1, -1)
        
        # Escalar datos
        features_scaled = scaler.transform(features)
        
        # Realizar predicción
        prediction = modelo.predict(features_scaled)[0]
        probability = modelo.predict_proba(features_scaled)[0][1]
        
        # Determinar nivel de confianza
        confidence = 'Alta' if abs(probability - 0.5) > 0.3 else 'Media'
        
        return jsonify({
            'prediccion': int(prediction),
            'probabilidad': float(probability),
            'confianza': confidence,
            'mensaje': 'Probable Internauta' if prediction == 1 else 'Probable No Internauta'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)