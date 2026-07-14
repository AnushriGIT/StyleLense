"""
main.py — Standalone FastAPI server for StyleLens Retrieval Core.
"""

import os
import io
import time
import uuid
import logging
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel

import retrieval_core as utils

'''
# ==========================================
# 1. SERVER SETUP & CONFIGURATION
# ==========================================
Import required libraries (FastAPI, PIL for images, Pydantic for validation)
Import retrieval_core (our machine learning models)
Set up a logger to track bugs and traffic

CREATE FastAPI Application Instance (the web server)

ENABLE CORS:
    Allow any website or frontend to talk to this server.

MOUNT Static Folder:
    IF the "static" folder exists on disk:
        Serve it at "http://.../static" so browsers can load CSS/JS images.
'''

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stylelens-retrieval")

app = FastAPI(
    title="StyleLens — Retrieval Core API",
    description="Layer 1: Deep-learning clothing image & text query similarity search.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
'''
app.mount() method attaches a completely independent application or service to a specific URL path on your main application.
'''

# Mount static folder if it exists
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Middleware: Request ID & Observability ────────────────────────────────────
'''
# ==========================================
# 2. MIDDLEWARE (Runs on EVERY request)
# ==========================================
INTERCEPT Every Incoming HTTP Request:
    Assign a unique "Request ID" to trace the user's action
    Start a timer
    Pass the request down into the actual endpoints below
    Stop the timer when the response comes back
    Log exactly how many milliseconds it took and whether it succeeded
    Return the response to the user
    '''
    
@app.middleware("http")
async def add_request_id_and_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start_time = time.time()
    logger.info(f"[{request_id}] Start {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(f"[{request_id}] Finished {request.method} {request.url.path} in {process_time:.2f}ms with status {response.status_code}")
    return response

# ── Endpoints ──────────────────────────────────────────────────────────────────

'''
# ==========================================
# 3. ENDPOINTS (The actual API routes)
# ==========================================
ROUTE GET "/health":
    # Used by monitoring systems to check if the AI is awake
    RETURN status, hardware, and database readiness

ROUTE POST "/search":
    # The main multi-modal search engine
    RECEIVE optional uploaded image AND/OR optional text query
    
    IF no image AND no text provided:
        THROW Error "Please provide an image or text"

    # Scenario A: Text Only
    IF ONLY text is provided:
        Search the CLIP database using just the text
        RETURN matches

    # Scenario B: Image Provided (with or without text)
    IF image is provided:
        Convert the image into an AI mathematical vector (Embedding)
        Find the closest matching clothes in the database
        Check the highest similarity score to assign confidence (high/medium/low)
        Automatically extract tags from the image (e.g. "red shirt", "stripes")

        # The "Pivot" Mechanism:
        IF confidence is "low" AND the user provided text AND we found tags:
            Assume visual search failed. Build a new text query using the tags + text
            Run a brand new text-based search instead.
            
        # The "Refinement" Mechanism:
        IF we didn't Pivot AND the user provided text:
            Re-rank the visual matches based on how well they match the text.

        RETURN the final list of matched clothes along with the confidence score.

ROUTE POST "/text-search":
    # Dedicated endpoint exclusively for text search
    RECEIVE a JSON request with a text string, search CLIP, return matches

ROUTE GET "/":
    # The Frontend Webpage
    Send the HTML code to the browser so the user sees a visual interface.
'''

class TextSearchRequest(BaseModel):
    """
    Pydantic model representing the request body for text-based clothing searches.
    
    Attributes:
        query (str): The natural language text describing the clothing item to search for.
        top_k (Optional[int]): The maximum number of similar matches to retrieve (default: 10).
    """
    query: str
    top_k: Optional[int] = 10

@app.get("/health")
def health():
    """Liveness check and model/gallery readiness status."""
    return {
        "status": "healthy",
        "device": str(utils.DEVICE),
        "resnet_gallery": {
            "ready": utils.RESNET_GALLERY is not None,
            "shape": list(utils.RESNET_GALLERY.shape) if utils.RESNET_GALLERY is not None else None
        },
        "clip_gallery": {
            "ready": utils.CLIP_GALLERY is not None,
            "shape": list(utils.CLIP_GALLERY.shape) if utils.CLIP_GALLERY is not None else None
        },
        "dinov2_gallery": {
            "ready": utils.DINOV2_GALLERY is not None,
            "shape": list(utils.DINOV2_GALLERY.shape) if utils.DINOV2_GALLERY is not None else None
        }
    }

@app.post("/search")
async def search(
    image: Optional[UploadFile] = File(None),
    user_query: Optional[str] = Form(None),
    top_k: int = Form(10)
):
    """
    Accepts an image upload (and optional text query) or a text query alone.
    Returns ranked similar clothes from the gallery with similarity scores and attributes.
    """
    if image is None and (user_query is None or not user_query.strip()):
        raise HTTPException(status_code=400, detail="Please provide an image or a text query.")

    try:
        # ── Text-only Mode ──────────────────────────────────────────────────────
        if image is None:
            query_str = user_query.strip()
            if utils.CLIP_GALLERY is None:
                raise HTTPException(status_code=503, detail="CLIP gallery embeddings database is not loaded.")
            candidates = utils.text_search(query_str, top_k)
            return {
                "mode": "text_search",
                "query": query_str,
                "confidence": "medium",
                "attribute_tags": [],
                "pivot_used": False,
                "pivot_query": None,
                "result_count": len(candidates),
                "results": candidates
            }

        # ── Image (with optional Text) Mode ─────────────────────────────────────
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Visual search via DINOv2
        if utils.DINOV2_GALLERY is None or utils.DINOV2_MODEL is None:
            raise HTTPException(status_code=503, detail="DINOv2 model or gallery embeddings database is not loaded.")
        
        query_emb = utils.embed_image_dinov2(pil_image)
        # Search the DINOv2 gallery to get the top 50 visual matches
        candidates = utils.nearest_search(query_emb, utils.DINOV2_GALLERY, top_k=50)
        top_score = candidates[0]["similarity"] if candidates else 0.0

        # Confidence categorization
        if top_score >= 78.0:
            confidence = "high"
        elif top_score >= 65.0:
            confidence = "medium"
        else:
            confidence = "low"

        # Zero-shot attribute tagging
        attribute_tags = utils._tag_attributes(pil_image)

        # Pivot logic for low confidence when query text is present
        pivot_used = False
        pivot_query = None
        query_text = user_query.strip() if user_query else ""

        if confidence == "low" and query_text and attribute_tags:
            pivot_query = " ".join(attribute_tags[:4])
            pivot_emb = utils.embed_text(pivot_query)
            candidates = utils.nearest_search(pivot_emb, utils.CLIP_GALLERY, top_k)
            pivot_used = True

        # Text refinement re-ranking if text is present and pivot wasn't triggered
        if query_text and not pivot_used:
            candidates = utils._text_refine(candidates, query_text)
            candidates = candidates[:top_k]
        elif not pivot_used:
            candidates = candidates[:top_k]

        return {
            "mode": "image_search",
            "confidence": confidence,
            "confidence_score": round(top_score, 1),
            "attribute_tags": attribute_tags,
            "pivot_used": pivot_used,
            "pivot_query": pivot_query,
            "result_count": len(candidates),
            "results": candidates
        }

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text-search")
def text_search_endpoint(request: TextSearchRequest):
    """CLIP text-only search against the clothing gallery."""
    if utils.CLIP_GALLERY is None:
        raise HTTPException(status_code=503, detail="CLIP gallery database is not loaded.")
    
    candidates = utils.text_search(request.query, request.top_k)
    return {
        "query": request.query,
        "result_count": len(candidates),
        "results": candidates
    }

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serves the single-page visual search interface."""
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
    if not os.path.exists(template_path):
        return HTMLResponse(content="<h3>StyleLens Demo Interface is not found. Check api/templates/index.html</h3>", status_code=404)
    with open(template_path) as f:
        return f.read()
