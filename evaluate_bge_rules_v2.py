import os
os.environ['HTTP_PROXY'] = 'http://192.168.81.170:7890'
os.environ['HTTPS_PROXY'] = 'http://192.168.81.170:7890'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import numpy as np

def clean_label(label):
    if not label: return '无'
    label = label.strip()
    if label == '' or '失败' in label: return '无'
    return label

def main():
    data_path = 'wyf-exp1/experiments/rewrite_io_20260313_231801.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    queries, labels = [], []
    for item in data:
        queries.append(item['original_query'])
        labels.append(clean_label(item.get('matched_rule')))

    # Combine queries and labels to remove duplicates
    unique_data = {}
    for q, l in zip(queries, labels):
        unique_data[q] = l
        
    uq = list(unique_data.keys())
    ul = list(unique_data.values())

    print(f'Original samples: {len(queries)}, Unique samples: {len(uq)}')
    print("\nLabel distribution in unique samples:")
    for k, v in Counter(ul).items():
        print(f"  {k}: {v}")
        
    le = LabelEncoder()
    y_encoded = le.fit_transform(ul)

    try:
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            uq, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    except ValueError:
        print("\nStratified split failed (probably a class has < 2 members). Falling back to random split.")
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            uq, y_encoded, test_size=0.2, random_state=42
        )

    print("\nLoading BGE model (BAAI/bge-large-zh-v1.5)...")
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    
    print("Encoding training data...")
    X_train = model.encode(X_train_text, show_progress_bar=True)
    print("Encoding testing data...")
    X_test = model.encode(X_test_text, show_progress_bar=True)

    print("\n" + "="*40)
    print("--- 1. Training MLP Classifier ---")
    clf_mlp = MLPClassifier(hidden_layer_sizes=(512, 128), max_iter=1000, random_state=42)
    clf_mlp.fit(X_train, y_train)
    y_pred_mlp = clf_mlp.predict(X_test)
    print(classification_report(y_test, y_pred_mlp, target_names=le.classes_, zero_division=0))
    print(f'MLP Overall Accuracy: {accuracy_score(y_test, y_pred_mlp):.4f}')

    print("\n" + "="*40)
    print("--- 2. Training SVM Classifier (Linear, balanced) ---")
    clf_svm = SVC(kernel='linear', class_weight='balanced', C=1.0, random_state=42)
    clf_svm.fit(X_train, y_train)
    y_pred_svm = clf_svm.predict(X_test)
    print(classification_report(y_test, y_pred_svm, target_names=le.classes_, zero_division=0))
    print(f'SVM Overall Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}')

    print("\n" + "="*40)
    print("--- 3. Training Logistic Regression (balanced) ---")
    clf_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf_lr.fit(X_train, y_train)
    y_pred_lr = clf_lr.predict(X_test)
    print(classification_report(y_test, y_pred_lr, target_names=le.classes_, zero_division=0))
    print(f'LR Overall Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}')

if __name__ == '__main__':
    main()
