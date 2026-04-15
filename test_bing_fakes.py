import os
import json
from pathlib import Path
from urllib.request import Request, urlopen
import uuid
import mimetypes

def _encode_multipart(fields: dict, file_field: str, file_path: Path):
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    file_name = file_path.name
    mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()

    parts = []
    for key, value in fields.items():
        parts.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
            str(value).encode("utf-8"),
            b"\r\n",
        ])
    parts.extend([
        f"--{boundary}\r\n".encode("utf-8"),
        f'Content-Disposition: form-data; name="{file_field}"; filename="{file_name}"\r\n'.encode("utf-8"),
        f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ])
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"

def evaluate_bing_fakes():
    bing_dir = Path("data/bing_fakes")
    with open(bing_dir / "metadata.json", "r") as f:
        samples = json.load(f)
        
    results = []
    print("--- Evaluating via Production API ---")
    total_fake_caught = 0
    for s in samples:
        if "filename" not in s: continue
        img_path = bing_dir / s["filename"]
        
        body, content_type = _encode_multipart({"model": "hybrid"}, "file", img_path)
        request = Request("http://127.0.0.1:8000/api/predict", data=body, headers={"Content-Type": content_type, "Content-Length": str(len(body))}, method="POST")
        try:
            with urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            
            pred_label = result.get("predicted_class", "unknown").lower()
            confidence = result.get("prediction_metrics", {}).get("hybrid_probability", 0.0)
            if "confidence" in result:
                confidence = result["confidence"]
                
            correct = (pred_label == s["label"])
            if correct: total_fake_caught += 1
            
            results.append({
                "filename": s["filename"],
                "source": s["source"],
                "true_label": s["label"],
                "predicted_label": pred_label,
                "confidence": confidence,
                "correct": correct
            })
            print(f"[file: {s['filename']}] True: {s['label']:4} -> Pred: {pred_label:4} (Conf: {confidence}) | Correct: {correct}")
        except Exception as e:
            print(f"Failed {s['filename']} - {e}")
            
    with open("reports/bing_evaluation.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Total caught: {total_fake_caught}/{len(samples)}")
        
if __name__ == "__main__":
    evaluate_bing_fakes()
