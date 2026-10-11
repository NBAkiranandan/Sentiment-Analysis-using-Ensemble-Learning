import requests, json

base = "http://localhost:8000"

tests = [
    ("GET", "/"),
    ("GET", "/api/health"),
    ("GET", "/api/dataset/stats"),
    ("GET", "/api/models"),
    ("GET", "/api/metrics"),
    ("GET", "/api/model-comparison"),
    ("GET", "/api/methodology"),
    ("GET", "/api/training/status"),
]

print("=== API ENDPOINT VALIDATION ===")
all_passed = True
for method, path in tests:
    try:
        r = requests.get(base + path, timeout=10)
        status = "PASS" if r.ok else "FAIL"
        if not r.ok:
            all_passed = False
        print(f"  [{status}] {method} {path:<30} => {r.status_code}")
    except Exception as e:
        all_passed = False
        print(f"  [ERROR] {method} {path:<30} => {e}")

print()
print("=== SINGLE PREDICTION TEST ===")
try:
    r1 = requests.post(base + "/api/predict", json={"text": "I absolutely love this amazing product!"}, timeout=10)
    d1 = r1.json()
    print(f"  Positive Test: Status={r1.status_code}")
    print(f"    Sentiment:   {d1['sentiment']}")
    print(f"    Confidence:  {d1['confidence']*100:.1f}%")
    print(f"    Model:       {d1['model']}")
    print(f"    Processed:   {d1['processed_text'][:60]}...")
    
    r2 = requests.post(base + "/api/predict", json={"text": "Terrible service, never buying again!"}, timeout=10)
    d2 = r2.json()
    print()
    print(f"  Negative Test: Status={r2.status_code}")
    print(f"    Sentiment:   {d2['sentiment']}")
    print(f"    Confidence:  {d2['confidence']*100:.1f}%")
    print(f"    Model:       {d2['model']}")
except Exception as e:
        print(f"  Prediction test failed: {e}")

print()
print("=== PREDICTION COMPARISON TEST (ALL 3 MODELS) ===")
try:
    rc = requests.post(base + "/api/predict/compare", json={"text": "I absolutely love this amazing experience!"}, timeout=10)
    dc = rc.json()
    print(f"  Compare Test: Status={rc.status_code}")
    print(f"    Text: {dc['text']}")
    print(f"    - Logistic Reg : {dc['logistic_regression']['sentiment']} ({dc['logistic_regression']['confidence']*100:.1f}%)")
    print(f"    - Random Forest: {dc['random_forest']['sentiment']} ({dc['random_forest']['confidence']*100:.1f}%)")
    print(f"    - AdaBoost     : {dc['adaboost']['sentiment']} ({dc['adaboost']['confidence']*100:.1f}%)")
except Exception as e:
    print(f"  Compare prediction test failed: {e}")

print()
print("=== METRICS SNAPSHOT ===")
try:
    r3 = requests.get(base + "/api/metrics", timeout=10)
    meta = r3.json()
    lr = meta.get("logistic_regression", {})
    rf = meta.get("random_forest", {})
    ada = meta.get("adaboost", {})
    print(f"  Logistic Reg:  acc={lr.get('accuracy', 0):.4f} prec={lr.get('precision', 0):.4f} rec={lr.get('recall', 0):.4f} f1={lr.get('f1_score', 0):.4f}")
    print(f"  Random Forest: acc={rf.get('accuracy', 0):.4f} prec={rf.get('precision', 0):.4f} rec={rf.get('recall', 0):.4f} f1={rf.get('f1_score', 0):.4f}")
    print(f"  AdaBoost:      acc={ada.get('accuracy', 0):.4f} prec={ada.get('precision', 0):.4f} rec={ada.get('recall', 0):.4f} f1={ada.get('f1_score', 0):.4f}")
    print(f"  Selected Model: {meta.get('selected_model')}")
except Exception as e:
    print(f"  Metrics test failed: {e}")

print()
print(f"=== OVERALL: {'ALL PASSED' if all_passed else 'SOME FAILED'} ===")
