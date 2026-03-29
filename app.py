from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import joblib 
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
if not app.secret_key:
    raise ValueError("No SECRET_KEY set for Flask application!")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.getenv("MODEL_DIR") or os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.getenv("MODEL_PATH") or os.path.join(MODEL_DIR, 'best_model.pkl')
SCALER_PATH = os.getenv("SCALER_PATH") or os.path.join(MODEL_DIR, 'scaler.pkl')
METADATA_PATH = os.getenv("METADATA_PATH") or os.path.join(MODEL_DIR, 'metadata.json')

try:
    best_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print('Model load error:', e)

with open(METADATA_PATH) as f:
    metadata = json.load(f)
    
FEATURE_COLS = metadata['feature_cols']

# Stage Info
check_mark = "\u2705"
cross_mark = "\u274C"
warning_mark = "\U000026A0\uFE0F"

STAGE_INFO = {
    0: {
        'name':'Normal',
        'color': '#10b981', 'icon': check_mark,
        'systolic': '<120', 'diastolic': '<80', 'risk': 'Low',
        'description': 'Routine annual checkup recommended',
        'urgency': 'Routine annual checkup recommended',
        'recommendations': [
            'Maintain regular physical activity (150+ min/week)',
            'Follow a balanced, low-sodium diet (DASH diet recommended)',
            'Monitor blood pressure annually',
            'Maintain healthy BMI (18.5–24.9)',
            'Limit alcohol consumption and avoid tobacco',
            'Manage stress through relaxation and mindfulness techniques'
            ]
        },
    1: {
        'name': 'Hypertension Stage 1',
        'color': '#f97316', 'icon': '\U0001F536',
        "systolic": '130-139', 'diastolic': '80-89',
        'risk': 'Moderate ',
        'description': 'Stage 1 Hypertension detected. Medical consultation and lifestyle changes are strongly recommended.',
        'urgency': 'Consult a physician within 1 month',
        'recommendations': [
            'Schedule appointment with your physician within 1 month',
            'Begin DASH diet with strict sodium restriction (< 1,500 mg/day)',
            'Monitor blood pressure daily — log morning and evening readings',
            'Structured exercise program (30 min/day, 5 days/week) with medical clearance',
            'Medication may be prescribed based on your cardiovascular risk profile',
            'Quit smoking immediately if applicable',
            'Reduce or eliminate alcohol consumption',
            'Achieve healthy weight if BMI > 25'
            ]
        },
    
    2: {
        'name': 'Hypertension Stage 2',
        'color': '#ef4444', 'icon': '\U0001F534',
        'systolic': '≥ 140', 'diastolic': '≥ 90',
        'risk': 'High ',
        'description': 'Stage 2 Hypertension — requires prompt medical attention. Antihypertensive medication is likely necessary.',
        'urgency': 'Urgent: See physician within 1 week',
        'recommendations': [
            'Seek medical attention within 1 week',
            'Antihypertensive medication is typically required at this stage',
            'Strict sodium restriction (< 1,500 mg/day)',
            'Daily blood pressure monitoring — morning and evening',
            'Absolute smoking cessation required',
            'Zero alcohol tolerance strongly recommended',
            'Supervised exercise program only with medical clearance',
            'Regular kidney function and cardiac monitoring',
            'Evaluate for secondary causes of hypertension'
        ]
        },
    3: {
        'name': 'Hypertensive Crisis',
        'color': '#dc2626', 'icon': '\U0001F6A8',
        'systolic': '≥ 180', 'diastolic': '≥ 120',
        'risk': 'Critical ',
        'description': 'HYPERTENSIVE CRISIS — This is a medical emergency requiring immediate intervention.',
        'urgency': 'EMERGENCY — Call emergency services immediately',
        'recommendations': [
            '🚨 CALL EMERGENCY SERVICES (108) IMMEDIATELY',
            'Do NOT drive yourself to the hospital',
            'Sit or lie down calmly while waiting for emergency services',
            'Take prescribed emergency medication only if instructed by a doctor',
            'Emergency IV antihypertensive therapy will be required',
            'Immediate organ damage assessment (ECG, CT scan, lab tests)',
            'ICU admission may be necessary',
            'Do NOT eat, drink, or take any unprescribed substances',
            'Stay calm and avoid any physical exertion'
        ]
    }
}

# Encode incoming form data to training encoding
def encode_input(data):
    """Map form values to exactly the same encoding used during training."""
    def b(v):
        return 1 if str(v).strip().lower() in ['yes', 'male', '1'] else 0
    
    age_map = {'0': 0, '1': 1, '2': 2, '3':3}
    severity_map = {'0': 0, '1': 1, '2': 2}
    diag_map = {'0': 0, '1': 1, '2': 2}
    systolic_map = {'0': 0, '1': 1, '2': 2, '3':3}
    diastolic_map = {'0': 0, '1': 1, '2': 2, '3':3, '4': 4}
    
    features = [
        b(data.get('gender', 0)),
        age_map.get(str(data.get('age', 0)), 0),
        b(data.get('history', 0)),
        b(data.get('patient', 0)),
        b(data.get('take_medication', 0)),
        severity_map.get(str(data.get('severity', 0)), 0),
        b(data.get('breath_shortness', 0)),
        b(data.get('visual_changes', 0)),
        b(data.get('nose_bleeding')),
        diag_map.get(str(data.get('when_diagnosed', 0)), 0),
        systolic_map.get(str(data.get('systolic', 0)), 0),
        diastolic_map.get(str(data.get('diastolic', 0)), 0),
        b(data.get('controlled_diet', 0))
    ]
    return np.array(features).reshape(1, -1)


@app.route('/')
def index():
    return render_template('index.html', metadata=metadata)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No input data'}), 400
        
        def safe_int(val, default=0):
            try:
                return int(val)
            except:
                return default
        features  = encode_input(data)
        if features.shape[1] != len(FEATURE_COLS):
            raise ValueError('Feature mismatch')
        features_df  = pd.DataFrame(features, columns=FEATURE_COLS)
        features_sc = scaler.transform(features_df)
        
        prediction = int(best_model.predict(features_sc)[0])
        stage = STAGE_INFO[prediction]
        
        # Probabilities
        probabilities = {}
        if hasattr(best_model, 'predict_proba'):
            proba = best_model.predict_proba(features_sc)[0]
            for i, p in enumerate(proba):
                probabilities[STAGE_INFO[i]['name']] = round(float(p) * 100, 1)
                
        else:
            probabilities = {stage['name']: 100.0}
            
        # Risk factors from submitted data
        risk_factors = []
        age_val = safe_int(data.get('age'))
        if age_val >= 3:
            risk_factors.append({'factor': 'Age 65+', 'impact': 'High'})
        elif age_val == 2:
            risk_factors.append({'factor': 'Age 51-64', 'impact': 'Moderate'})
            
        if safe_int(data.get('history')) == 1:
            risk_factors.append({'factor': 'Family History of HTN', 'impact': 'High'})
        if safe_int(data.get('patient')) == 1:
            risk_factors.append({'factor': 'Existing Patient (HTN)', 'impact': 'High'})
        if str(data.get('severity', 0)) == '2':
            risk_factors.append({'factor': 'Severe Symptoms', 'impact': 'High'})
        elif str(data.get('severity', 0)) == '1':
            risk_factors.append({'factor': 'Moderate Symptoms', 'impact': 'Moderate'})
        if safe_int(data.get('breath_shortness', 0)) == 1:
            risk_factors.append({'factor': 'Shortness of Breath', 'impact': 'High'})
        if safe_int(data.get('visual_changes', 0)) == 1:
            risk_factors.append({'factor': 'Visual Changes', 'impact': 'High'})
        if safe_int(data.get('nose_bleeding', 0)) == 1:
            risk_factors.append({'factor': 'Nose Bleeding', 'impact': 'Moderate'})
        if safe_int(data.get('take_medication', 0)) == 0:
            risk_factors.append({'factor': 'Not Taking Medication', 'impact': 'Moderate'})
        if safe_int(data.get('controlled_diet', 0)) == 0:
            risk_factors.append({'factor': 'Uncontrolled Diet', 'impact': 'Low'})
        if str(data.get('when_diagnosed', 0)) == '2':
            risk_factors.append({'factor': 'HTN for >5 Years', 'impact': 'High'})
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'stage_map': stage['name'],
            'stage_color': stage['color'],
            'stage_icon': stage['icon'],
            'risk_level': stage['risk'],
            'description': stage['description'],
            'urgency': stage['urgency'],
            'recommendations': stage['recommendations'],
            'probabilities': probabilities,
            'risk_factors': risk_factors,
            'bp_range': {
                'systolic': stage['systolic'],
                'diastolic': stage['diastolic']
            },
            'model_used': metadata['best_model'],
            'model_accuracy': metadata['model_results'][metadata['best_model']]['accuracy']
        })    
        
    except Exception as e:
        print('ERROR:', e)
        return jsonify({'success': False, 'error': str(e)}), 500
    
    
@app.route('/model-info')
def model_info():
    return jsonify({
        'best_model': metadata['best_model'],
        'model_results': metadata['model_results'],
        'feature_importance': metadata.get('feature_importance', {}),
        'dataset_info': metadata['dataset_info'],
        'overfitting_analysis': metadata.get('overfitting_analysis', {})
        })
    
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': metadata['best_model']})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)