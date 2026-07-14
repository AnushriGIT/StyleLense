# System Patterns & Architecture

This document describes the architectural flow, middleware design, and mathematical patterns governing the StyleLens retrieval engine.

---

## 1. Request ID & Observability Pipeline

Every HTTP request to the API is intercepted by a custom middleware layer (`add_request_id_and_logging`) which enforces tracing and diagnostics:

```
[Incoming Request] ──> [Extract/Gen Request ID] ──> [Start Stopwatch] ──> [Execute Endpoint]
                                                                                │
[Client Response] <── [Inject ID into Headers] <── [Log Process Time] <── [Capture Response]
```

---

## 2. Core Search Mathematical Patterns

### Vector Dot-Products as Cosine Similarity
All embedding extractors (ResNet, CLIP) apply $L_2$-normalization to raw vectors before outputs:
$$v_{norm} = \frac{v}{\max(\|v\|_2, 10^{-8})}$$

Because vectors are scaled to unit length (magnitude of $1.0$), we calculate **Cosine Similarity** using a single matrix multiplication in Numpy instead of expensive loops:
```python
scores = query_emb @ gallery_emb.T
```

---

## 3. Multimodal Logic Flow (`POST /search`)

*   **Scenario A (Text Only):** Text encoded via CLIP text encoder $\rightarrow$ cosine search against CLIP gallery.
*   **Scenario B (Image Only):** Image encoded via CLIP vision encoder $\rightarrow$ cosine search against CLIP gallery.
*   **Refinement Re-ranking:** If both Image and Text are present, candidate image scores are refined:
    $$\text{Score}_{combined} = 0.7 \times \text{Similarity}_{visual} + 0.3 \times \text{Similarity}_{text}$$
*   **Zero-Shot Tagging & Pivot:** 
    *   Images are classified against a fixed `ATTRIBUTE_VOCABULARY` using CLIP.
    *   If visual similarity confidence is **low** (< 65.0%) and query text is present, the engine automatically pivots: it constructs a new query combining the text and the top predicted visual tags, then runs a fresh text-only search.
