from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Cargar modelo y scaler
modelo = joblib.load('modelo_internauta_logreg.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del request
        data = request.get_json()
        
        # Crear array con características
        features = np.array([
            float(data['edad']),
            float(data['renta']),
            float(data['region']),
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