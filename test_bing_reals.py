import os
import json
import joblib
from utils.classical_features import extract_classical_features_from_path, ClassicalFeatureSpec

def evaluate_bing_reals():
    meta_path = "data/bing_reals/metadata.json"
    if not os.path.exists(meta_path):
        print("No metadata.json found.")
        return
        
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    correct = 0
    total = len(metadata)
    
    print("--- Evaluating Bing Real Targets Natively ---")
    clf = joblib.load("models/classical_fallback_model.joblib")
    with open("models/classical_fallback_metadata.json", "r") as f:
        m = json.load(f)
    threshold = m["decision_threshold"]
    spec = ClassicalFeatureSpec()
    
    predictions = []
    
    for item in metadata:
        img_path = os.path.join("data/bing_reals", item["filename"])
        if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
            print(f"Skipping {item['filename']}")
            total -= 1
            continue
            
        try:
            vec = extract_classical_features_from_path(img_path, spec)
            prob_fake = clf.predict_proba(vec.reshape(1, -1))[0, 0]
            prob_real = clf.predict_proba(vec.reshape(1, -1))[0, 1]
            pred_label = "real" if prob_real >= threshold else "fake"
            conf = max(prob_real, prob_fake)
        except Exception as e:
            print(f"File fault {item['filename']}: {e}")
            continue

        is_correct = (pred_label == "real")
        if is_correct:
            correct += 1
            
        predictions.append({
            "filename": item["filename"],
            "correct": is_correct
        })
        
        print(f"[file: {item['filename']}] True: real -> Pred: {pred_label} (Conf real: {prob_real:.4f}) | Correct: {is_correct}")

    print(f"\nFinal: {correct}/{total} Real Images Correctly Managed.")

if __name__ == "__main__":
    evaluate_bing_reals()
