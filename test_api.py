'''
API Test Suite
Tests all flask routes, prediction logic, overfitting analysis, 
model artifacts, dataset integrity.

'''
import pytest
import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

# Fixtures

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client
        
@pytest.fixture
def normal_patient():
    return {
        'gender': 'female', 'age': '0',
        'history': '0', 'patient': '0',
        'when_diagnosed': '0', 'severity': '0',
        'breath_shortness': '0', 'visual_changes': '0',
        'nose_bleeding': '0', 'systolic': '1',
        'diastolic': '0', 'take_medication': '0',
        'controlled_diet': '1'
    }
    
@pytest.fixture
def stage1_patient():
    return {
       'gender': 'male', 'age': '0',
        'history': '0', 'patient': '0',
        'when_diagnosed': '0', 'severity': '1',
        'breath_shortness': '1', 'visual_changes': '0',
        'nose_bleeding': '0', 'systolic': '2',
        'diastolic': '1', 'take_medication': '0',
        'controlled_diet': '0' 
    }
    
@pytest.fixture
def stage2_patient():
    return {
        'gender': 'male', 'age': '2',
        'history': '1', 'patient': '1',
        'when_diagnosed': '1', 'severity': '1',
        'breath_shortness': '1', 'visual_changes': '0',
        'nose_bleeding': '0', 'systolic': '2',
        'diastolic': '2', 'take_medication': '1',
        'controlled_diet': '0'
    }
    
@pytest.fixture
def crisis_patient():
    return {
        'gender': 'male', 'age': '3',
        'history': '1', 'patient': '1',
        'when_diagnosed': '2', 'severity': '2',
        'breath_shortness': '1', 'visual_changes': '1',
        'nose_bleeding': '1', 'systolic': '3',
        'diastolic': '4', 'take_medication': '0',
        'controlled_diet': '0'
    }
    
# Test Group 1: Route Availability

class TestRoutes:

    def test_homepage_returns_200(self, client):
        r = client.get('/')
        assert r.status_code == 200
        
    def test_health_returns_200(self, client):
        r = client.get('/health')
        assert r.status_code == 200
        
    def test_health_status_ok(self, client):
        r = client.get('/health')
        d = json.loads(r.data)
        assert d['status'] == 'ok'

    def test_model_returns_200(self, client):
        r = client.get('/model-info')
        assert r.status_code == 200
        
    def test_predict_rejects_get_method(self, client):
        r = client.get('/predict')
        assert r.status_code == 405
        
    def test_unknown_route_returns_404(self, client):
        r = client.get('/nonexistent-route')
        assert r.status_code == 404
       
# Test Group 2: Overfitting Analysis 
class TestOverfittingAnalysis:
    # Model Selection Logic
    def test_logistic_regression_is_selected(self, client):
        r = client.get('/model-info')
        d = json.loads(r.data)
        assert d['model_results']['Logistic Regression']['status'] == 'Selected'
        
    def test_decision_tree_is_overfitted(self, client):
        r = client.get('/model-info')
        d = json.loads(r.data)
        assert d['model_results']['Decision Tree']['generalization'] == 'Overfitted'
        
    def test_exactly_1_model_selected(self, client):
        r = client.get('/model-info')
        d = json.loads(r.data)
        selected = [n for n, v in d['model_results'].items() if v['status'] == 'Selected']
        assert len(selected) == 1
        
# Test Group 3: Prediction Corretness
class TestPredictions:
        
    def test_predict_success_flag(self, client, stage2_patient):
        r = client.post('/predict',
                       data=json.dumps(stage2_patient),
                       content_type='application/json')
        d = r.get_json()
        assert r.status_code == 200
        
        assert d is not None, f"Response body was empty! Status: {r.status_code}"
        assert d['success'] is True
        
    def test_predict_returns_valid_stage(self, client, stage2_patient):
        r = client.post('/predict',
                       data=json.dumps(stage2_patient),
                       content_type='application/json')
        d = json.loads(r.data)
        assert d['prediction'] in [0, 1, 2, 3]
        
    def test_predict_has_all_required_fields(self, client, stage2_patient):
        r = client.post('/predict',
                       data=json.dumps(stage2_patient),
                       content_type='application/json')
        d = json.loads(r.data)
        for field in ['prediction', 'stage_map', 'stage_color', 'stage_icon',
                      'risk_level', 'description', 'urgency', 'recommendations',
                      'probabilities', 'risk_factors', 'bp_range',
                      'model_used', 'model_accuracy']:
            assert field in d, f"missing: {field}"
            
    def test_predict_probabilities_sum_to_100(self, client, stage2_patient):
        r = client.post('/predict',
                       data=json.dumps(stage2_patient),
                       content_type='application/json')
        d = json.loads(r.data)
        total = sum(d['probabilities'].values())
        assert abs(total - 100.0) < 1.0
        
    def test_predict_has_4_probability_classes(self, client, stage2_patient):
        r = client.post('/predict', json=stage2_patient)
        d = r.get_json()
        assert r.status_code == 200, f"Server error: {r.data.decode()}"
        assert 'probabilities' in d, 'Probabilities missing from response'
        assert len(d['probabilities']) == 4
        
    def test_predict_bp_range_has_systolic_diastolic(self, client, stage2_patient):
        r  = client.post('/predict',
                        data=json.dumps(stage2_patient),
                        content_type='application/json')
        d = json.loads(r.data)
        assert 'systolic' in d['bp_range']
        assert 'diastolic' in d['bp_range']
        
    def test_crisis_patient_risk_is_critical(self, client, crisis_patient):
        r = client.post('/predict',
                       data=json.dumps(crisis_patient),
                       content_type='application/json')
        d = json.loads(r.data)
        assert d['risk_level'] == 'Critical '
        
    def test_crisis_prediction_is_stage_3(self, client, crisis_patient):
        r = client.post('/predict',
                       data=json.dumps(crisis_patient),
                       content_type='application/json')
        d = json.loads(r.data)
        assert  d['prediction'] == 3
        
    # Response Quality
    def test_predict_recommendations_not_empty(self, client, stage2_patient):
        r = client.post('/predict',
                       data=json.dumps(stage2_patient),
                       content_type='application/json')
        d = json.loads(r.data)
        assert len(d['recommendations']) >= 1
        
    def test_stage_name_is_string(self, client, stage2_patient):
        r = client.post('/predict',
                       data=json.dumps(stage2_patient),
                       content_type='application/json')
        d = json.loads(r.data)
        assert isinstance(d['stage_map'], str)
        assert len(d['stage_map']) > 0
        
    def test_risk_factors_is_list(self, client, stage2_patient):
        r = client.post('/predict',
                       data=json.dumps(stage2_patient),
                       content_type='application/json')
        d = json.loads(r.data)
        assert isinstance(d['risk_factors'], list)
        
        
        
        
# Test Group 4: Data Integrity
class TestDataset:
   
    def test_patient_data_exists(self):
        assert os.path.exists('data/patient_data.csv'), \
            'patient_data.csv missing from data/'
            
    def test_train_plus_test_equals_1825(self):
       train = pd.read_csv('data/hypertension_train.csv')
       test = pd.read_csv('data/hypertension_test.csv')
       assert len(train) + len(test) == 1825
       
    def test_no_null_in_train(self):
        df = pd.read_csv('data/hypertension_train.csv')
        assert df.isnull().sum().sum() == 0
        
    def test_train_stages_are_0_to_3(self):
        df = pd.read_csv('data/hypertension_train.csv')
        assert set(df['Stages'].unique()).issubset({0, 1, 2, 3})
        
    # Dataset Structure
    def test_train_has_all_columns(self):
        df = pd.read_csv('data/hypertension_train.csv')
        required = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication',
                    'Severity', 'BreathShortness', 'VisualChanges', 'NoseBleeding',
                    'Whendiagnoused', 'Systolic', 'Diastolic', 'ControlledDiet', 'Stages']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_train_has_13_feature_columns(self):
        df = pd.read_csv('data/hypertension_train.csv', index_col=0)
        assert len(df.columns) == 14  # 13 features + 1 target
        
# Test Group 5:  Model Artifact Integrity
class TestArtifacts:
    # Metadata Validation
    def test_metadata_best_model(self):
        with open('models/metadata.json') as f:
            m = json.load(f)
        assert m['best_model'] == 'Logistic Regression'
        
    def test_metadata_has_13_feature_cols(self):
        with open('models/metadata.json') as f:
            m = json.load(f)
        assert len(m['feature_cols']) == 13
    
    def test_metadata_total_samples   (self):
        with open('models/metadata.json') as f:
            m = json.load(f)
            assert m['dataset_info']['total_samples'] == 1825
    
    def test_best_model_pkl_exists(self):
        assert os.path.exists("models/best_model.pkl")
        
    def test_scaler_pkl_exists(self):
        assert os.path.exists("models/scaler.pkl")
        
    def test_metadata_json_exists(self):
        assert os.path.exists("models/metadata.json")

# Run Directly

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
    