import os
import requests
import json
import time

urls = [
    "https://thumbs.dreamstime.com/b/real-person-face-smiling-woman-portrait-blue-background-34700615.jpg",
    "https://thumbs.dreamstime.com/b/smiling-portrait-man-face-real-retro-colour-high-detail-32390610.jpg",
    "https://thumbs.dreamstime.com/b/real-man-face-portrait-looking-camera-blue-background-33714577.jpg",
    "https://thumbs.dreamstime.com/b/real-man-face-portrait-looking-camera-blue-background-33713265.jpg",
    "https://st2.depositphotos.com/1011382/7491/i/950/depositphotos_74911405-stock-photo-real-man-face-looking-at.jpg",
    "https://thumbs.dreamstime.com/b/real-man-face-portrait-looking-camera-blue-background-33713025.jpg",
    "https://thumbs.dreamstime.com/b/real-man-face-portrait-looking-camera-blue-background-33714859.jpg",
    "https://thumbs.dreamstime.com/b/smiling-portrait-man-face-real-retro-colour-high-detail-32390572.jpg",
    "https://thumbs.dreamstime.com/b/real-man-face-portrait-looking-camera-blue-background-33714666.jpg",
    "https://thumbs.dreamstime.com/b/real-man-face-portrait-looking-camera-blue-background-33714743.jpg",
    "https://c8.alamy.com/comp/KYXR92/real-people-face-KYXR92.jpg",
    "https://c8.alamy.com/comp/KYXR77/real-people-face-KYXR77.jpg",
    "https://thumbs.dreamstime.com/b/real-man-face-portrait-looking-camera-blue-background-33718597.jpg",
    "https://c8.alamy.com/comp/KYXR82/real-people-face-KYXR82.jpg",
    "https://c8.alamy.com/comp/KYYPJ0/real-people-face-KYYPJ0.jpg",
    "https://thumbs.dreamstime.com/z/real-person-face-smiling-woman-portrait-blue-background-34700593.jpg"
]

os.makedirs("data/bing_reals", exist_ok=True)
metadata = []

for i, url in enumerate(urls):
    ext = url.split('.')[-1]
    if len(ext) > 4:
        ext = "jpg"
    filename = f"bing_real_{i}.{ext}"
    out_path = os.path.join("data/bing_reals", filename)
    
    print(f"Downloading {filename}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            with open(out_path, 'wb') as f:
                f.write(resp.content)
            metadata.append({
                "filename": filename,
                "url": url,
                "source": url.split('/')[2]
            })
            print(f" -> Saved {out_path}")
        else:
            print(f" -> Failed with {resp.status_code}")
    except Exception as e:
        print(f" -> Error: {e}")
    time.sleep(0.5)

with open("data/bing_reals/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Successfully downloaded {len(metadata)} images.")
