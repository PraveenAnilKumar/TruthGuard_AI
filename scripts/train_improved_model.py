"""
train_improved_model.py - Improved classical model with better balance
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime
import os

print("="*70)
print("🎯 IMPROVED FAKE NEWS DETECTOR - Balanced Training")
print("="*70)

# Load data
print("\n📂 Loading dataset...")
df = pd.read_csv('D:\\TruthGuard_AI\\datasets\\fake_news\\all_fake_news_combined.csv')
texts = df['text'].astype(str).tolist()
labels = df['label'].tolist()

real_count = labels.count(0)
fake_count = labels.count(1)
print(f"📊 Total samples: {len(texts)}")
print(f"📊 Class distribution: REAL={real_count}, FAKE={fake_count}")
print(f"📊 Imbalance ratio: 1:{fake_count/real_count:.2f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Create TF-IDF vectorizer with more features
vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words='english',
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.8,
    sublinear_tf=True  # Use 1+log(tf) instead of raw tf
)

# Transform text to features
print("\n🔧 Creating feature vectors...")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"📊 Feature matrix shape: {X_train_vec.shape}")

# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"⚖️ Class weights: REAL={class_weights[0]:.3f}, FAKE={class_weights[1]:.3f}")

# Try multiple models and pick the best
models = {
    'LogisticRegression': LogisticRegression(
        C=2.0,
        max_iter=2000,
        class_weight='balanced',
        random_state=42,
        solver='liblinear'
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=50,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
}

best_model = None
best_score = 0
best_model_name = ""

print("\n🔍 Training multiple models...")

for name, model in models.items():
    print(f"\n📈 Training {name}...")
    model.fit(X_train_vec, y_train)
    
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test_vec)
    y_pred = model.predict(X_test_vec)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Use F1 score as the primary metric (balance of precision/recall)
    if f1 > best_score:
        best_score = f1
        best_model = model
        best_model_name = name
        best_y_pred = y_pred
        best_proba = y_pred_proba

print(f"\n🏆 Best model: {best_model_name} (F1-Score: {best_score:.4f})")

# Find optimal threshold
print("\n🎚️ Finding optimal prediction threshold...")
thresholds = np.arange(0.3, 0.8, 0.05)
best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    y_pred_thresh = (best_proba[:, 1] >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"✅ Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")

# Final evaluation with optimal threshold
y_pred_final = (best_proba[:, 1] >= best_threshold).astype(int)

print("\n📊 Final Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=['REAL', 'FAKE']))

print("\n📉 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_final)
print(cm)
print(f"True Real: {cm[0][0]}, False Fake: {cm[0][1]}")
print(f"False Real: {cm[1][0]}, True Fake: {cm[1][1]}")

accuracy = (cm[0][0] + cm[1][1]) / len(y_test)
print(f"\n✅ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Create pipeline with best model
final_pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('classifier', best_model)
])

# Save model and metadata
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"D:\\TruthGuard_AI\\models\\improved_model_{timestamp}"
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, 'fake_news_model.pkl')
joblib.dump(final_pipeline, model_path)

# Save metadata
metadata = {
    'model_type': best_model_name,
    'accuracy': float(accuracy),
    'f1_score': float(best_score),
    'optimal_threshold': float(best_threshold),
    'class_weights': class_weight_dict,
    'training_date': timestamp,
    'samples': len(texts)
}

import json
with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n💾 Model saved to: {save_dir}")
print("✅ Training complete!")