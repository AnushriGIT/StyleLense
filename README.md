# 🛍️ StyleLens: Multi-Modal Fashion Retrieval Engine
> **DeepFashion Cross-Domain Clothing Image & Text Search Core**

StyleLens is a state-of-the-art multi-modal fashion retrieval system that solves the **cross-domain clothing discovery problem**: matching uncurated consumer mobile photos (or natural language queries) to studio catalogue product images.

---

## 🌟 Executive & Business Overview

In real-world fashion e-commerce, users upload phone photos with arbitrary lighting, complex backgrounds, and varied angles. Standard zero-shot models fail to retrieve the correct catalogue items because they match background noise rather than garment geometry (yielding a baseline Recall@1 of **3.38%**).

StyleLens bridges this domain gap using fine-tuned Vision Transformers (**CLIP ViT-B/32** and **DINOv2 ViT-B/14**), achieving a **+390% accuracy boost (16.58% Recall@1 / 46.53% Recall@20)** and serving sub-10ms similarity searches (**~8.2 ms**) across catalogue items.

---

## 📐 System Architecture & Workflow

```mermaid
flowchart TD
    subgraph Data Processing Pipeline
        DF["DeepFashion Benchmark Dataset"] --> EDA["01_eda.ipynb"]
        EDA --> DS["fashion_ds_processing.py\n(Streaming PyTorch Loaders & Hard Negative Sampler)"]
    end

    subgraph Training & Modeling Engine
        DS --> baseline["02_model_experiments.ipynb\n(ResNet-50 / EfficientNet / DINOv2 / CLIP)"]
        DS --> ft_clip["03_fine_tuning.ipynb & train_clip_vlm_hybrid.py\n(Triplet Loss + InfoNCE + Hard Negatives)"]
        DS --> siamese["04_siamese.ipynb\n(Siamese ResNet Re-Ranker)"]
    end

    subgraph Evaluation & Benchmarking
        ft_clip --> eval["scripts/evaluate_retrieval.py\n(Recall@1, 5, 10, 20 Benchmarking)"]
    end

    subgraph Production Inference Engine
        eval --> API["FastAPI Serving Layer\n(api/main.py & retrieval_core.py)"]
        API --> UI["Web Dashboard UI\n(http://127.0.0.1:8000/)"]
    end
```

---

## ⚡ Core System Modules & Technical Highlights

| Core System Module | Primary Engineering Focus | Key Technical Components | Code & Implementation |
| :--- | :--- | :--- | :--- |
| **Exploratory Data Analysis (EDA)** | DeepFashion dataset analytics, domain gap & category long-tail distributions. | Analysis of consumer vs shop domain gap, image resolution variance, annotation distributions. | [`notebooks/01_eda.ipynb`](file:///Users/anushrin/Documents/AG/ShopAssist/StyleLense/notebooks/01_eda.ipynb) |
| **Data Engineering & Sampling** | Streaming PyTorch DataLoader pipeline with hard-negative mining. | Category-constrained negative sampler, PIL image streaming, VLM caption collation. | [`fashion_ds_processing.py`](file:///Users/anushrin/Documents/AG/ShopAssist/StyleLense/fashion_ds_processing.py) |
| **Multi-Backbone Modeling Engine** | Comparative fine-tuning across CNNs, Vision Transformers & CLIP. | ResNet-50, EfficientNet-B4, DINOv2 ViT-B/14, OpenAI CLIP ViT-B/32, Triplet Loss, InfoNCE. | [`notebooks/02_model_experiments.ipynb`](file:///Users/anushrin/Documents/AG/ShopAssist/StyleLense/notebooks/02_model_experiments.ipynb)<br>[`notebooks/03_fine_tuning.ipynb`](file:///Users/anushrin/Documents/AG/ShopAssist/StyleLense/notebooks/03_fine_tuning.ipynb)<br>[`notebooks/04_siamese.ipynb`](file:///Users/anushrin/Documents/AG/ShopAssist/StyleLense/notebooks/04_siamese.ipynb) |
| **Multi-Modal Search Engine** | 5-stage ablation & natural language text query optimization. | Hybrid loss ($\mathcal{L}_{\text{triplet}} + \mathcal{L}_{\text{InfoNCE}}$), differential LRs, attribute dropout ($p=0.20$). | [`scripts/train_clip_vlm_hybrid.py`](file:///Users/anushrin/Documents/AG/ShopAssist/StyleLense/scripts/train_clip_vlm_hybrid.py) |
| **Production Serving Layer** | Asynchronous REST API & web dashboard for sub-10ms search. | FastAPI, Uvicorn, L2 vector normalization, cosine dot-product search, 90 Pytest unit tests. | [`api/main.py`](file:///Users/anushrin/Documents/AG/ShopAssist/StyleLense/api/main.py)<br>[`retrieval_core.py`](file:///Users/anushrin/Documents/AG/ShopAssist/StyleLense/retrieval_core.py)<br>[`tests/`](file:///Users/anushrin/Documents/AG/ShopAssist/StyleLense/tests/) |

---

## 🏆 Key Experimental Results (Recall@K Benchmarking)

| Model Architecture & Variant | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Technical Highlights |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-50 (Baseline)** | 3.08% | 5.69% | 7.37% | 9.70% | Single-domain ImageNet baseline |
| **ResNet-50 (Fine-Tuned)** | 5.94% | 12.41% | 16.42% | 21.51% | Triplet fine-tuned ResNet |
| **EfficientNet-B4 (Baseline)** | 3.89% | 7.27% | 9.39% | 12.08% | Zero-shot CNN baseline |
| **EfficientNet-B4 (Fine-Tuned)** | 8.18% | 16.61% | 21.46% | 27.30% | Triplet fine-tuned EfficientNet |
| **CLIP ViT-B/32 (Baseline)** | 3.38% | 5.77% | 7.10% | 8.77% | Zero-shot OpenAI CLIP baseline |
| **CLIP ViT-B/32 (Fine-Tuned, Image-Only)**| **16.58%** | **31.59%** | **38.62%** | **46.53%** | **+390% gain** over zero-shot |
| **DINOv2 ViT-B/14 (Baseline)** | 3.62% | 6.24% | 7.76% | 9.62% | Zero-shot Facebook DINOv2 baseline |
| **DINOv2 ViT-B/14 (Fine-Tuned)** 🏆 | **16.80%** | **31.98%** | **39.10%** | **46.52%** | **Best Pure Visual Matcher** |
| **CLIP Hybrid Run C (+ Hard Negatives)** | **15.50%** | **30.90%** | **38.50%** | **47.00%** | Category hard negative visual peak |
| **CLIP Text Attempt 3 (Deep Text+Vision)** 🏆| **14.50%** | **29.10%** | **36.50%** | **45.10%** | **Deployed Production Champion (Image + Text)** |

---

## ⚡ Installation & Quick Start Guide

### 1. Prerequisites & Virtual Environment Setup
Ensure Python 3.10+ is installed on your machine.
```bash
# Clone repository
git clone https://github.com/<your-username>/StyleLense.git
cd StyleLense

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Verification
Run the environment verification script to check dataset paths and PyTorch/Torchvision installations:
```bash
python setup_check.py
```

### 3. Run Automated Pytest Suite
Execute all 90 automated unit tests covering API endpoints, dataset processing, and vector search logic:
```bash
.venv/bin/pytest tests/
```

### 4. Start Live FastAPI Web Server
Launch the asynchronous Uvicorn server:
```bash
PYTHONPATH=. .venv/bin/uvicorn api.main:app --reload --port 8000
```
Open your browser and navigate to:
- **Interactive UI**: `http://127.0.0.1:8000/`
- **OpenAPI Swagger Docs**: `http://127.0.0.1:8000/docs`

---

## 💡 API Usage & Sample Requests

### 1. Image Search (`POST /search`)
Send a clothing image to retrieve the top $K$ matching shop items:
```bash
curl -X POST "http://127.0.0.1:8000/search" \
  -F "image=@/path/to/consumer_photo.jpg" \
  -F "top_k=5"
```
**Sample JSON Response**:
```json
{
  "mode": "image_search",
  "confidence": "high",
  "result_count": 5,
  "results": [
    {
      "rank": 1,
      "similarity": 88.42,
      "image_path": "/path/to/shop_01.jpg",
      "image_data": "/9j/4AAQSkZJRgABAQ..."
    }
  ]
}
```

### 2. Text Search (`POST /text-search`)
Send a natural language query to retrieve matching clothing items:
```bash
curl -X POST "http://127.0.0.1:8000/text-search" \
  -F "query=white formal girls top with lace work" \
  -F "top_k=5"
```

---

## 📁 Repository Directory Structure

```text
StyleLense/
├── README.md                         # Main repository documentation & setup guide
├── config.py                         # Path resolution, embedding dims, batch size settings
├── requirements.txt                  # Python dependencies
├── setup_check.py                    # Environment and dataset path verification
├── fashion_ds_processing.py          # PyTorch datasets (Triplet, InfoNCE, Hard Negative Sampler)
├── retrieval_core.py                 # Core embedding helpers, L2 normalization & vector search
├── notebooks/                        # Project Notebooks
│   ├── 01_eda.ipynb                  # EDA, class distribution, resolution variance, domain gap
│   ├── 02_model_experiments.ipynb    # ResNet-50, EfficientNet-B4, DINOv2, CLIP baselines
│   ├── 03_fine_tuning.ipynb          # Triplet loss fine-tuning & unfreeze layer experiments
│   └── 04_siamese.ipynb              # Contrastive loss Siamese standalone & re-ranker
├── scripts/                          # Automated Scripts
│   ├── train_clip_vlm_hybrid.py      # CLI script for 5-stage ablation & text search fine-tuning
│   ├── extract_embeddings.py         # Batch embedding extraction script
│   └── evaluate_retrieval.py         # Automated Recall@K evaluation harness
├── api/                              # Serving Layer
│   ├── main.py                       # FastAPI application (/search, /text-search, /health)
│   └── static/, templates/           # Interactive Web UI dashboard
├── Reference/                        # Technical Documentation & Comparison Reports
│   └── ModelComparison_DINOv2_RunD_RunE_TextAttempt3.md
└── tests/                            # Pytest automated test suite (90 unit tests)
```

---

## 🔮 Future Enhancements & Scope Extensions

1. **Vector Database Integration**: Upgrade exact L2 NumPy matrix dot product search to **FAISS GPU / Milvus / Qdrant** if indexing catalogue sizes above 10,000,000 items.
2. **Multilingual Search**: Extend text encoder using multilingual vision-language models (e.g., **XLM-RoBERTa / M-CLIP**) to support global multi-language retail queries.
3. **Wardrobe & Multi-Image Input**: Support multi-image composition queries (e.g. combining top + bottom photos for outfit recommendation).
