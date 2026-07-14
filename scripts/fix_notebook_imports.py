import json

NB_PATH = "/Users/anushrin/Documents/AG/ShopAssist/StyleLense/notebooks/03_fine_tuning.ipynb"

with open(NB_PATH, "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "from datasets import FashionDataset" in source:
            if "sys.path.append" not in source:
                new_source = "import sys\nimport os\nsys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))\n" + source
                
                # Convert back to lines
                lines = []
                for line in new_source.split('\n')[:-1]:
                    lines.append(line + "\n")
                if new_source and not new_source.endswith('\n'):
                    lines.append(new_source.split('\n')[-1])
                
                cell["source"] = lines
                print("Added sys.path.append to the imports cell.")
            else:
                print("sys.path.append already present.")
            break

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook path patched successfully!")
