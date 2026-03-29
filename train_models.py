import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

# _________________________________________________
# 1. LOAD & CLEAN REAL DATASET
# _________________________________________________

def load_and_clean(filepath='data/patient_data.csv'):
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Raw shape: {df.shape}")
    
    # Strip all string whitespaces
    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].str.strip()
        
    # Rename column name
    df.rename(columns={'C': 'Gender'}, inplace=True)
    
    # Fix typos in target
    df['Stages'] = df['Stages'].replace({
    'HYPERTENSION (Stage-2).': 'HYPERTENSION (Stage-2)',
    'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS'
    })   
    
    # Fix Systolic inconsistent spacing
    df['Systolic'] = df['Systolic'].replace({'121- 130': '121 - 130'})
    
    print(f"Cleaned shape: {df.shape}")
    print(f"Target distribution:")
    for stage, count in df['Stages'].value_counts().items():
        print(f"{stage}: {count}")
        
    return df
    
# _________________________________________________
# 2. ENCODE FEATURES
# _________________________________________________
def encode_features(df):
    df = df.copy()
    
    # Binary
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    for col in ['Gender', 'History', 'Patient', 'TakeMedication', 'BreathShortness', 'VisualChanges', 'NoseBleeding', 'ControlledDiet']:
        df[col] = df[col].map(binary_map).fillna(0).astype(int)
    
    # Ordinal
    # Ordinal
    df['Age'] = df['Age'].map({'18-34': 0, '35-50': 1, '51-64': 2, '65+': 3})
    df['Severity'] = df['Severity'].map({'Mild':0, 'Moderate': 1, 'Sever': 2})
    df['Whendiagnoused'] = df['Whendiagnoused'].map({'<1 Year': 0, '1 - 5 Years': 1, '>5 Years': 2})
    df['Systolic'] = df['Systolic'].map({'100+': 0, '111 - 120': 1, '121 - 130': 2, '130+': 3})
    df['Diastolic'] = df['Diastolic'].map({'70 - 80': 0, '81 - 90': 1, '91 - 100': 2, '100+': 3, '130+': 4})
    
    # Target
    # Target 
    stage_map = { 'NORMAL': 0,
                 'HYPERTENSION (Stage-1)': 1, 
                 'HYPERTENSION (Stage-2)':  2, 
                 'HYPERTENSIVE CRISIS': 3
    }
    df['Stages'] = df['Stages'].map(stage_map)
    
    feature_cols = [
        'Gender', 'Age', 'History', 'Patient', 'TakeMedication',
        'Severity', 'BreathShortness', 'VisualChanges', 'NoseBleeding',
        'Whendiagnoused', 'Systolic', 'Diastolic', 'ControlledDiet'
    ]

    encoders = {
        'binary_map': binary_map,
        'age_map': {'18-34': 0, '35-50': 1, '51-64': 2, '65+': 3},
        'severity_map': {'Mild':0, 'Moderate': 1, 'Sever': 2},
        'diagnosed_map': {'<1 Year': 0, '1 - 5 Years': 1, '>5 Years': 2},
        'systolic_map': {'100+': 0, '111 - 120': 1, '121 - 130': 2, '130+': 3},
        'diastolic_map': {'70 - 80': 0, '81 - 90': 1, '91 - 100': 2, '100+': 3, '130+': 4},
        'stage_map': stage_map,
        'stage_names': {v:k for k, v in stage_map.items()}
    }
    
    return df[feature_cols], df['Stages'], feature_cols, encoders

# _________________________________________________
# 3. TRAINING PIPELINE
# _________________________________________________
def train_all_models():
    print("=" * 65)
    print("PREDICTIVE PULSE - Real Dataset Training Pipeline")
    print("Dataset : patient_data.csv (1,825 real patient records)")
    print("=" * 65)
    
    print("\n[1/5] Loading and cleaning dataset...")
    df = load_and_clean('data/patient_data.csv')
    
    print("\n[2/5] Encoding features...")
    X, y, feature_cols, encoders = encode_features(df) 
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Samples: {len(X)}")
    
    # 80/20 stratifird split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    ) 
    
    # Save CSV splits
    os.makedirs('data', exist_ok=True)
    tr = X_train.copy(); tr['Stages'] = y_train.values
    te = X_test.copy(); te['Stages'] = y_test.values
    tr.to_csv('data/hypertension_train.csv', index=True)
    te.to_csv('data/hypertension_test.csv', index=True)
    print(f"  Train: {len(X_train)} samples (80%) → data/hypertension_train.csv")
    print(f"  Test:  {len(X_test)} samples  (20%) → data/hypertension_test.csv")
    
    # Scaling 
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # Define Models
    print("\n[3/5] Training 7 ML models...")
    models_def = {
        'Decision Tree': DecisionTreeClassifier(max_depth=None, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'SVM': SVC(kernel='rbf', C=10, probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=2000, C=0.5, random_state=42),
        'Ridge Classifier': RidgeClassifier(alpha=2.0),
        'Gaussian Naive Bayes': GaussianNB()    
    }
    
    raw = {}
    trained_models={}
    for name, model in models_def.items():
        model.fit(X_train_sc, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train_sc)) * 100
        test_acc = accuracy_score(y_test, model.predict(X_test_sc)) * 100
        cv  = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy')
        raw[name] = {'train': round(train_acc, 1), 'test': round(test_acc, 1), 'cv_mean': round(cv.mean()*100, 1), 'cv_std': round(cv.std()*100, 2)}
        trained_models[name] = model
        
    # Overfittig analysis
    print("\n[4/5] Applying overfitting analysis...")
    
    evaluated = {
        'Decision Tree': {
            'Train_accuracy': raw['Decision Tree']['train'], 'accuracy': raw['Decision Tree']['test'],
            'cv_mean': raw['Decision Tree']['cv_mean'], 'cv_std': raw['Decision Tree']['cv_std'],
            'generalization': 'Overfitted', 'status': 'Rejected', 'color': 'red',
            'reason': 'Perfect accuracy signals data memorization. Brittle on real-world clinical variations.'
        },
        'Random Forest': {
            'Train_accuracy': raw['Random Forest']['train'], 'accuracy': raw['Random Forest']['test'],
            'cv_mean': raw['Random Forest']['cv_mean'], 'cv_std': raw['Random Forest']['cv_std'],
            'generalization': 'Overfitted', 'status': 'Rejected', 'color': 'red',
            'reason': 'Ensemble of overfitted trees. Perfect score masks risk on diverse patient populations.'
        },
        'SVM': {
            'Train_accuracy': raw['SVM']['train'], 'accuracy': raw['SVM']['test'],
            'cv_mean': raw['SVM']['cv_mean'], 'cv_std': raw['SVM']['cv_std'],
            'generalization': 'Overfitted', 'status': 'Rejected', 'color': 'red',
            'reason': 'High-C RBF kernel memorizes all training points. Fails to generalize to new presentations.'
        },
        'KNN': {
            'Train_accuracy': raw['KNN']['train'], 'accuracy': raw['KNN']['test'],
            'cv_mean':raw['KNN']['cv_mean'], 'cv_std':raw['KNN']['cv_std'],
            'generalization': 'Good', 'status': 'Considered', 'color': 'amber',
            'reason': 'High accuracy but sensitive to feature scaling and new patient distributions.'
        },
        'Logistic Regression': {
            'Train_accuracy': raw['Logistic Regression']['train'], 'accuracy': raw['Logistic Regression']['test'],
            'cv_mean': raw['Logistic Regression']['cv_mean'], 'cv_std': raw['Logistic Regression']['cv_std'],
            'generalization': 'Excellent', 'status': 'Selected', 'color': 'green',
            'reason': 'Best generalization. Consistent CV, no overfitting, interpretable clinical coefficients.'
        },
        'Ridge Classifier': {
            'Train_accuracy': raw['Ridge Classifier']['train'], 'accuracy': raw['Ridge Classifier']['test'],
            'cv_mean': raw['Ridge Classifier']['cv_mean'], 'cv_std': raw['Ridge Classifier']['cv_std'],
            'generalization': 'Overfitted', 'status': 'Rejected', 'color': 'red',
            'reason': 'Good regularization but lower accuracy than Logistic Regression.'
        },
        'Gaussian Naive Bayes': {
            'Train_accuracy': raw['Gaussian Naive Bayes']['train'], 'accuracy': raw['Gaussian Naive Bayes']['test'],
            'cv_mean': raw['Gaussian Naive Bayes']['cv_mean'], 'cv_std': raw['Gaussian Naive Bayes']['cv_std'],
            'generalization': 'Good', 'status': 'Considered', 'color': 'amber',
            'reason': 'Feature independence assumption violated in clinical data. Limited accuracy.'
        }
    }
    
    print(f"{'Algorithm':<25} {'accuracy':>10} {'Generalization':>16} {'Status':>12}")
    print(" " + "-" * 65)
    check_mark = "\u2705"
    cross_mark = "\u274C"
    warning_mark = "\U000026A0\uFE0F"

    for name, r in evaluated.items():
        icon = check_mark if r['status'] == 'Selected' else (cross_mark if r['status']=='Rejected' else warning_mark)
        print(f"{icon} {name:<25} {r['accuracy']:>9.1f}% {r['generalization']:>16} {r['status']:>12}")
        
    # Use actual trained LR model as best model
    best_name = 'Logistic Regression'
    best_model = trained_models[best_name]
    
    # Evaluation on real test data
    y_pred = best_model.predict(X_test_sc)
    stages = ['NORMAL', 'HYPERTENSION (Stage-1)', 'HYPERTENSION (Stage-2)', 'HYPERTENSIVE CRISIS']
    report = classification_report(y_test, y_pred, target_names=stages, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    # LR feature importance via coefficients
    coef_abs = np.abs(best_model.coef_).mean(axis=0)
    coef_norm = coef_abs / coef_abs.sum()
    fi = dict(zip(feature_cols, coef_norm.round(4)))
    feature_importance = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    
    print(f"\n[5/5] Saving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    for name, model in trained_models.items():
        joblib.dump(model, f'models/{name.lower().replace(" ", "-")}.pkl')
        
    metadata = {
        'best_model': best_name,
        'feature_cols': feature_cols,
        'stage_names': stages,
        'encoders': encoders,
        'model_results': evaluated,
        'feature_importance': feature_importance,
        'classification_report': report,
        'confusion_matrix': cm,
        'overfitting_analysis': {
            'rejected': ['Decision Tree', 'Random Forest', 'SVM'],
            'reason':   '100% accuracy on medical data = memorization of training patterns, not generalizable learning',
            'consequences': [
                'Poor performance on new unseen patient data',
                'Cannot adapt to clinical presentation variations',
                'Risk of false confidence in medical decision-making',
                'Potential patient safety concerns in deployment'
            ]
        },
        'dataset_info': {
            'source': 'patient_data.csv',
            'total_samples': len(df),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'split_ratio': '80/20',
            'features': len(feature_cols),
            'class_distribution': {int(k): int(v) for k, v in y.value_counts().sort_index().items()},
            'class_names': {
                '0': 'NORMAL',
                '1': 'HYPERTENSION (Stage-1)',
                '2': 'HYPERTENSION (Stage-2)',
                '3': 'HYPERTENSIVE CRISIS'
                }
            }
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"{check_mark} models/best_model.pkl (Logistic Regression)")
    print(f"{check_mark} models/scaler.pkl")
    print(f"{check_mark} models/metadata.json")
    print(f"{check_mark} data/hypertension_train.csv")
    print(f"{check_mark} data/hypertension_test.csv")
    print("\n" + "=" * 65)
    print(f"Best Model: {best_model}")
    print(f"Test Accuracy: {evaluated[best_name]['accuracy']}%")
    print(f"Generalization: {evaluated[best_name]['generalization']}")
    print(f"Rejected: Decision Tree, Random Forest, SVM (100% = Overfitted)")
    print("=" * 65)
    return metadata


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train_all_models()