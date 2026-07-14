import json
import re

NB_PATH = "/Users/anushrin/Documents/AG/ShopAssist/StyleLense/notebooks/03_fine_tuning.ipynb"

with open(NB_PATH, "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
        
    source = "".join(cell.get("source", []))
    
    # 1. Remove the SKIP exceptions from ALL cells
    source = re.sub(r'# === SKIP:.*?===\nraise Exception\("SKIP"\)\n*', '', source)
    
    # 2. Make ResNet training graceful
    if "EPOCHS     = 5" in source and "resnet50_ft.train()" in source:
        # Wrap the whole block in an if-else
        new_source = """import os

best_model_path = os.path.join(MODELS_DIR, "resnet50_best.pth")
if os.path.exists(best_model_path):
    print("✅ Found fine-tuned ResNet-50 weights. Skipping training and loading weights.")
    resnet50_ft.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
else:
"""
        # Indent the original source
        for line in source.split('\n'):
            if line.strip():
                new_source += "    " + line + "\n"
            else:
                new_source += "\n"
        source = new_source

    # 3. Make CLIP training graceful
    elif "EPOCHS     = 5" in source and "clip_model.train()" in source:
        new_source = """import os

best_model_path = os.path.join(MODELS_DIR, "clip_best.pth")
if os.path.exists(best_model_path):
    print("✅ Found fine-tuned CLIP weights. Skipping training and loading weights.")
    clip_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
else:
"""
        for line in source.split('\n'):
            if line.strip():
                new_source += "    " + line + "\n"
            else:
                new_source += "\n"
        source = new_source

    # 4. Make DINOv2 training graceful
    elif "EPOCHS     = 5" in source and "dinov2_ft.train()" in source:
        new_source = """import os

best_model_path = os.path.join(MODELS_DIR, "dinov2_best.pth")
if os.path.exists(best_model_path):
    print("✅ Found fine-tuned DINOv2 weights. Skipping training and loading weights.")
    dinov2_ft.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
else:
"""
        for line in source.split('\n'):
            if line.strip():
                new_source += "    " + line + "\n"
            else:
                new_source += "\n"
        source = new_source

    # Convert source back to list of lines for Jupyter JSON format
    lines = []
    for line in source.split('\n')[:-1]: # last element from split is empty if it ended in \n
        lines.append(line + "\n")
    if source and not source.endswith('\n'):
        lines.append(source.split('\n')[-1])
        
    cell["source"] = lines

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully!")
