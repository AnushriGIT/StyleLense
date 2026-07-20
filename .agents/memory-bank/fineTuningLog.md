# Fine-Tuning & Evaluation Log

This log records model fine-tuning runs, hyperparameters, and resulting Recall@K scores to maintain durable memory.

## Run: clipImage&Text Hybrid Fine-Tuning
* **Date**: 2026-07-19
* **Commit Hash**: `c16e85e212f6fa8523dd83f124315cfa8e725b1b`
* **Model Type**: CLIP ViT-B/32 (`openai/clip-vit-base-patch32`)
* **Training Strategy**: Hybrid loss with triplet dominant and InfoNCE regularizer.
* **Freeze Strategy**: Freeze vision model except for the last 2 encoder layers, projection heads. Trainable parameters: 14.8M (9.80%).

### Hyperparameters
* **Epochs**: 5
* **Batch Size**: 16
* **Learning Rate**: 5e-6
* **Triplet Margin**: 0.2
* **Loss Weights Schedule**:
  * **Epoch 1**: $\lambda_{\text{triplet}} = 0.0$, $\lambda_{\text{contrastive}} = 1.0$ (Warmup)
  * **Epoch 2**: $\lambda_{\text{triplet}} = 0.5$, $\lambda_{\text{contrastive}} = 0.25$
  * **Epoch 3–5**: $\lambda_{\text{triplet}} = 1.0$, $\lambda_{\text{contrastive}} = 0.25$

### Evaluation Results (Recall@K)

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-50 (baseline)** | 3.08% | 5.69% | 7.37% | 9.70% |
| **ResNet-50 (fine-tuned)** | 5.94% | 12.41% | 16.42% | 21.51% |
| **EfficientNet-B4 (baseline)** | 3.89% | 7.27% | 9.39% | 12.08% |
| **EfficientNet-B4 (fine-tuned)** | 8.18% | 16.61% | 21.46% | 27.30% |
| **CLIP ViT-B/32 (baseline)** | 3.38% | 5.77% | 7.10% | 8.77% |
| **CLIP ViT-B/32 (fine-tuned, image-only)** | 16.58% | 31.59% | 38.62% | 46.53% |
| **DINOv2 ViT-B/14 (baseline)** | 3.62% | 6.24% | 7.76% | 9.62% |
| **DINOv2 ViT-B/14 (fine-tuned)** | 16.80% | 31.98% | 39.10% | 46.52% |
| **CLIP Image & Text (hybrid fine-tuned, 2-block)** | 13.21% | 26.29% | 32.96% | 40.70% |
| **CLIP Image & Text (hybrid fine-tuned, 3-block)** | 12.57% | 25.40% | 31.78% | 39.41% |
| **Siamese (ResNet-50 standalone)** | 2.22% | 5.37% | 7.98% | 11.63% |
| **Siamese (ResNet-50 re-ranker)** | 2.80% | 6.80% | 9.00% | 10.60% |
| **CLIP Text-Image (shop_captions.json, 5% subset)** | 3.30% | 5.60% | 6.90% | 8.60% |

*(Image-only fine-tuned CLIP row added 2026-07-19 during reconciliation with `clipHybridFineTuningPlan.md` — the table previously jumped straight from CLIP baseline to the hybrid result, omitting the image-only variant already recorded in `projectDocs.md`. All three CLIP variants — base, image-only fine-tuned, image+text hybrid fine-tuned — now sit together for direct comparison.)*

*(3-block hybrid row added 2026-07-19: a controlled re-run of the hybrid training with the vision freeze depth matched to the image-only CLIP/DINOv2 precedent elsewhere in this notebook — last 3 ViT blocks unfrozen instead of 2, everything else identical (same LR, margin, schedule, data). Result: Recall@1 **dropped** from 13.21% to 12.57% — the opposite of the expected direction. This refutes the "trainable capacity was the bottleneck" hypothesis; see `clipHybridFineTuningPlan.md` for the full writeup and what this implies instead.)*

## Run: ResNet-50 Siamese Network Optimization
* **Date**: 2026-07-20
* **Model Type**: Siamese ResNet-50 with 256-dim projection head.
* **Freeze Strategy**: Full backbone load with projection head fine-tuning.
* **Loss Strategy**: Contrastive margin loss with same-category constrained negative sampling.
* **Transforms**: $224 \times 224$ ImageNet transforms.
* **Results Summary**: Underperforms CLIP and fine-tuned ResNet-50 backbone. Standalone Recall@1 is **2.22%**; candidate re-ranking increases Recall@1 to **2.80%** and Recall@10 to **9.00%** (outperforming the ResNet-50 baseline at higher K values, but failing to reach CLIP's performance).

## Run: CLIP Text-Image using shop_captions.json (Full Training)
* **Date**: 2026-07-20
* **Model Type**: CLIP ViT-B/32
* **Training Strategy**: Symmetric InfoNCE Loss between shop images and text descriptions from `shop_captions.json`.
* **Hyperparameters**: 3 Epochs, batch size 32, LR 1e-5. Run on the full 100% training dataset (~98k images).
* **Freeze Strategy**: Frozen text and vision encoders except for the last two layers.
* **Results Summary**: R@1: 1.9%, R@5: 3.1%, R@10: 3.8%, R@20: 4.9%. 
* **Observation**: The full training run performed *worse* than both the zero-shot baseline (3.38%) and the 5% 1-epoch subset test (3.30%). This regression is typical in contrastive fine-tuning when the batch size (32) is too small to provide enough negative samples per batch for InfoNCE, or when the learning rate causes catastrophic forgetting in the unfrozen layers. Larger batch sizes (e.g. 256+) or a triplet margin loss are likely required to see improvements.
