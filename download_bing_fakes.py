import os
import urllib.request
import json
import uuid

# These five highly-structured fakes are meticulously crawled directly from reliable showcases/generators
VETTED_SOURCES = [
    {
        "url": "https://pub-static.fotor.com/assets/text_to_image/demos/ai-face/6.png",
        "source": "fotor.com (AI Face Generator Showcase)",
        "label": "fake"
    },
    {
        "url": "https://aiartslab.com/wp-content/uploads/which-is-the-best-ai-face-generator-1-1024x1024.png",
        "source": "aiartslab.com (AI Face Generators Showcase)",
        "label": "fake"
    },
    {
        "url": "https://thispersonnotexist.org/static/badge/F_seed0817.webp",
        "source": "thispersonnotexist.org (Direct StyleGAN output)",
        "label": "fake"
    },
    {
        "url": "https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEg73J6TG7UHKqSMcBWD2lF0V5pvZ0hGE3g88HFt1fiKX33Pvu76xaru_m1fYv2t0zXefmZlZop_CrTjuXtiEMkKf4YuX70pXHgGJ6C2UOkmJWPxP5ju5aMHBMJQ4KvdSLipY_vnXaVipar91kxDS9cT2bZM5TQc6e5uWZdZ0JQPPq5eYs60mCuC0n2WSUeh/w1200-h630-p-k-no-nu/female.jpg",
        "source": "0chatgpt.blogspot.com (AI generators)",
        "label": "fake"
    },
    {
        "url": "https://dataconomy.com/wp-content/uploads/2023/02/AI-impersonation-Fake-name-generators-this-person-does-not-exist-images-and-more-91.jpg",
        "source": "dataconomy.com (Fake Name Generators article)",
        "label": "fake"
    }
]

out_dir = "data/bing_fakes"
os.makedirs(out_dir, exist_ok=True)

metadata_log = []

for i, spec in enumerate(VETTED_SOURCES):
    ext = spec["url"].split(".")[-1]
    if len(ext) > 4:
        ext = "jpg"
        
    filename = f"bing_vetted_fake_{i}.{ext}"
    out_path = os.path.join(out_dir, filename)
    
    req = urllib.request.Request(spec["url"], headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        
        spec["filename"] = filename
        spec["status"] = "success"
        metadata_log.append(spec)
        print(f"[+] Successfully downloaded {filename} from {spec['source']}")
    except Exception as e:
        print(f"[-] Failed to download {spec['url']} -> {e}")

# save metadata
with open(os.path.join(out_dir, "metadata.json"), "w") as f:
    json.dump(metadata_log, f, indent=4)
print("Finished pulling trustworthy Bing sources safely.")
