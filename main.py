
import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay,RocCurveDisplay
from imblearn.pipeline import Pipeline as ImbPipeline
import shap
from sklearn.ensemble import  ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from imblearn.over_sampling import SMOTE
#%%
def extract_features(image_path):
    img = cv2.imread(image_path, 0)
    if img is None: return None
    img = cv2.resize(img, (128, 128)) 

    # ميزات الملمس LBP
    lbp = local_binary_pattern(img, P=24, R=3, method='uniform')
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    # ميزات التردد FFT
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    (fft_hist, _) = np.histogram(magnitude_spectrum.ravel(), bins=30, range=(0, 255))
    fft_hist = fft_hist.astype("float")
    fft_hist /= (fft_hist.sum() + 1e-7)

    # دمج الميزات
    combined_features = np.hstack([lbp_hist, fft_hist])
    return combined_features
features = []
labels = []

data_paths = {
    r'natural_images': 0,        # الأصلي
    r'simulated_recaptured': 1   # المُحاكى
}
categories = ['Original', 'Recaptured']
print("جاري استخراج الميزات من الصور")

for path, label in data_paths.items():
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                full_path = os.path.join(root, file)
                f = extract_features(full_path)
                if f is not None:
                    features.append(f)
                    labels.append(label)

X = np.array(features)
y = np.array(labels)
#%%
# ==========================================
# Split Data
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"تم تحميل {len(X_train)} صورة بنجاح.")
print(f"شكل البيانات: {X_train.shape}")
print(np.unique(y_train, return_counts=True))

# ==========================================
# Define Models
# ==========================================
model_params = {
    "XGBoost": {
        "model": XGBClassifier(eval_metric='logloss', random_state=42),
        "params": {
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.01, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0]
        }
    },
    
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "model__max_depth": [5, 10, None],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        }
    },
    "ExtraTrees": {
        "model": ExtraTreesClassifier(random_state=42),
        "params": {
            "model__n_estimators": [100, 200] 
        }
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "params": {
            "model__C": [1, 10],
            "model__kernel": ["rbf"] 
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "model__n_neighbors": [3, 5, 7]
        }
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "model__C": [0.1, 1, 10]
        }
    },
    "Naive Bayes": {
    "model": GaussianNB(),
    "params": {
        "model__var_smoothing": [1e-9, 1e-8, 1e-7]
    }
},
}

best_estimators = {}
results = {}
print("\n--- Training Models ---")

for name, mp in model_params.items():
    pipeline = ImbPipeline([ 
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', mp["model"])
    ])
    grid = GridSearchCV(
        pipeline,
        mp["params"],
        cv=5,
        scoring="balanced_accuracy",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_estimators[name] = grid.best_estimator_
    results[name] = grid.best_score_
    print(f"{name}: {grid.best_score_*100:.2f}%")
#%%    
# ==========================================
# Select Winner
# ==========================================
winner_name = max(results, key=results.get)
final_model = best_estimators[winner_name]
print(f"\nWinner Model: {winner_name}")

# ==========================================
# Final Evaluation
# ==========================================
y_pred = final_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nFinal Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=categories))
# Training Accuracy
y_train_pred = final_model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"Train Accuracy: {train_acc*100:.2f}%")

# ==========================================
# Confusion Matrix
# ==========================================
ConfusionMatrixDisplay.from_estimator(
    final_model,
    X_test,
    y_test,
    display_labels=categories
)

plt.title(f"Confusion Matrix - {winner_name}")
plt.show()

# ==========================================
#RocCurveDisplay
# ==========================================
RocCurveDisplay.from_estimator(final_model, X_test, y_test)
plt.title(f"ROC Curve: {winner_name}")
plt.show()
gap = train_acc - acc
print(f"Generalization Gap: {gap*100:.2f}%")
#%%
# ==========================================
# XAI (SHAP) - النسخة المرتبة للأهمية الكلية
# ==========================================

X_train_summary = shap.sample(X_train, 40) 
explainer = shap.KernelExplainer(final_model.predict_proba, X_train_summary)

X_test_subset = shap.sample(X_test, 20)
shap_values = explainer.shap_values(X_test_subset, nsamples=100)

# SHAP  
if isinstance(shap_values, list):
    vals = np.abs(shap_values[1])
else:
    vals = np.abs(shap_values)
    if vals.ndim == 3:
        vals = vals[:, :, 1]

mean_shap = np.mean(vals, axis=0)

feature_names = [f'LBP_{i}' for i in range(26)] + [f'FFT_{i}' for i in range(30)]

if len(mean_shap) != len(feature_names):
    feature_names = [f'Feat_{i}' for i in range(len(mean_shap))]

# الترتيب والرسم
indices = np.argsort(mean_shap)
top_n = 20 

plt.figure(figsize=(10, 8))
plt.barh(range(top_n), mean_shap[indices][-top_n:], color='dodgerblue', align='center')
plt.yticks(range(top_n), [feature_names[i] for i in indices[-top_n:]])
plt.xlabel("Mean |SHAP Value|")
plt.title(f"Top {top_n} Features - {winner_name}")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
#%%
# ==========================================
# Save Model
# ==========================================
joblib.dump({
    "model": final_model,
    "categories": categories,
    "model_name": winner_name,
    "X_train": X_train[:50],
    "scores": {k: v * 100 for k, v in results.items()},
}, "model.pkl")

