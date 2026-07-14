# Web Architecture Notes: Flask vs FastAPI vs Frontend

> **Context:** Reference notes from building the StyleLens production serving layer.
> Covers when to use Flask vs FastAPI, frontend choices when calling an AI agent, and what CORS is.

---

### Flask vs FastAPI — When to Use Which

**Flask (current setup):**
- Simple, synchronous, easy to reason about
- Great for serving HTML + handling model inference for a single user at a time
- Dev server blocks on each request — fine for demos, not for production traffic
- Not suitable for concurrent users or long-running agent calls

**FastAPI:**
- Built on `asyncio` — handles many requests concurrently without blocking
- Native `StreamingResponse` — agent can stream tokens back word by word
- WebSocket support — real-time back-and-forth conversation
- Auto-generated `/docs` (Swagger UI) for free
- Every route needs `async def`; PyTorch inference must be offloaded to avoid blocking:
  ```python
  await loop.run_in_executor(None, embed_image, img)
  ```
- Request/response bodies need Pydantic type models (more boilerplate upfront)
- Different startup: `uvicorn step5_app:app --reload`

**Rule of thumb:**

| Scenario | Right Choice |
|---|---|
| Demo / single user | Flask |
| Agent with single response | Flask with longer timeout, or FastAPI |
| Agent with streaming output | FastAPI + StreamingResponse (SSE) |
| Conversational chat agent | FastAPI + WebSocket |

---

### Frontend Choice When Calling an Agent

**Plain HTML + `fetch()`**  
Right for: query → single agent response. Simple, no build tooling.

**Plain HTML + `EventSource` (SSE)**  
Right for: streaming agent response — agent types back word by word.  
~20 lines of vanilla JS, no framework needed.

**React / Vue**  
Right for: multi-turn chat with history, tool call results shown inline, intermediate agent steps visualized live.  
Adds npm + webpack/vite build step — only worth it when state complexity outgrows vanilla JS.

For this capstone: current HTML frontend is correct for fashion search.  
If adding an agent layer → migrate to FastAPI + keep HTML (add SSE if streaming).  
React only if building a full chat interface with visible reasoning steps.

---

### CORS — What It Is and When It Matters

CORS (Cross-Origin Resource Sharing) is a browser security rule: JavaScript on page A cannot call an API on server B unless server B explicitly allows it via a response header:

```
Access-Control-Allow-Origin: *
```

`flask_cors` adds this header automatically to every Flask response.

Not strictly needed when frontend and backend share the same Flask origin.  
Required if you split them — e.g. React dev server on `:3000` calling Flask API on `:5000`.
