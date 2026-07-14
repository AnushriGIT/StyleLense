# Project Documentation & Domain Knowledge

StyleLens is a deep-learning clothing retrieval and similarity search engine (Layer 1 Core API). It allows shoppers to search for clothing items using either text descriptions, uploaded images, or a combination of both.

---

## 1. Data & Model Assets

### Models:
*   **ResNet-50 (Visual Baseline):** Used for pure image embedding extraction (outputs a 2048-dim vector).
*   **CLIP (ViT-B/32, Multimodal):** Used for text queries, visual queries, and zero-shot tagging (outputs a 512-dim vector). Fine-tuned weights are saved under `models/clip_best/` (containing `config.json` and `model.safetensors`).
*   **DINOv2 (ViT-B/14, Candidate):** Evaluated as a potential visual backbone. Demonstrates a superior fine-tuned Recall@1 of **16.80%** (outperforming ResNet-50's 5.94%). Note: DINOv2 is a vision-only model and lacks text capability.

### Databases (Galleries):
*   Pre-computed vector embeddings for the clothing gallery are stored in `embeddings/` as `.npy` matrices:
    *   `trained_resnet50_gallery.npy`
    *   `trained_clip_gallery.npy` (or base `clip_gallery.npy`)
*   *Critical Rule:* If any model weights are updated/fine-tuned, the gallery `.npy` files must be re-generated to prevent dimensional mismatch.

### Dataset & License
Training/evaluation images and annotations come from **DeepFashion** (Consumer-to-shop Clothes Retrieval Benchmark), Multimedia Lab, The Chinese University of Hong Kong: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

> Please note that we do not own the copyrights to these images. Their use is **RESTRICTED to non-commercial research and educational purposes**.

If used in a publication, cite:
```bibtex
@inproceedings{liu2016deepfashion,
  author = {Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang},
  title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
  booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = June,
  year = {2016}
}
```

---

## 2. Benchmark Evaluations (Recall@K)

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
| :--- | :--- | :--- | :--- | :--- |
| ResNet-50 (baseline) | 3.08% | 5.69% | 7.37% | 9.70% |
| ResNet-50 (fine-tuned) | 5.94% | 12.41% | 16.42% | 21.51% |
| EfficientNet-B4 (baseline) | 3.89% | 7.27% | 9.39% | 12.08% |
| EfficientNet-B4 (fine-tuned) | 8.18% | 16.61% | 21.46% | 27.30% |
| CLIP ViT-B/32 (baseline) | 3.38% | 5.77% | 7.10% | 8.77% |
| DINOv2 ViT-B/14 (baseline) | 3.62% | 6.24% | 7.76% | 9.62% |
| **DINOv2 ViT-B/14 (fine-tuned)** | **16.80%** | **31.98%** | **39.10%** | **46.52%** |
