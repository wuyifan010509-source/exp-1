import os
os.environ['HTTP_PROXY'] = 'http://192.168.81.170:7890'
os.environ['HTTPS_PROXY'] = 'http://192.168.81.170:7890'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer
import numpy as np

def clean_label(label):
    if not label:
        return "无"
    label = label.strip()
    if label == "":
        return "无"
    return label

def main():
    data_path = 'wyf-exp1/experiments/rewrite_io_20260313_230620.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare data
    queries = []
    labels = []
    
    for item in data:
        queries.append(item['original_query'])
        labels.append(clean_label(item.get('matched_rule')))
        
    print(f"Total samples: {len(queries)}")
    print("Label distribution:")
    for k, v in Counter(labels).items():
        print(f"  {k}: {v}")
        
    # Split into train and test sets (80-20 stratified)
    # Filter out classes with too few samples to allow stratification, or just use regular split
    # Since some classes might be very small, we might need to be careful with stratification.
    # Actually rule 6 has 8 samples, 80% is 6, 20% is 2. It should be fine.
    try:
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            queries, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except ValueError:
        print("Stratified split failed (maybe a class has only 1 member), falling back to random split.")
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            queries, labels, test_size=0.2, random_state=42
        )
        
    print(f"\nTrain size: {len(X_train_text)}, Test size: {len(X_test_text)}")
    
    # Save train and test sets
    train_data = [{"original_query": q, "matched_rule": l} for q, l in zip(X_train_text, y_train)]
    test_data = [{"original_query": q, "matched_rule": l} for q, l in zip(X_test_text, y_test)]
    
    with open('wyf-exp1/experiments/train_split.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open('wyf-exp1/experiments/test_split.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        
    print("Train and test splits saved.")

    # Load BGE model
    print("\nLoading BAAI/bge-large-zh-v1.5 model...")
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    
    print("Encoding training data...")
    X_train = model.encode(X_train_text, show_progress_bar=True)
    print("Encoding testing data...")
    X_test = model.encode(X_test_text, show_progress_bar=True)
    
    # Train classifier
    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

if __name__ == '__main__':
    main()
