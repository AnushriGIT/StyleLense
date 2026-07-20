# StyleLens Tech Stack Reference

This document provides a brief reference for the key server technologies used in the StyleLens project.

## Web Server Layer

### FastAPI
- **What it is**: A modern, high-performance web framework for building APIs with Python 3.8+ based on standard Python type hints.
- **Key Features**:
  - Extremely fast (on par with NodeJS and Go).
  - Out-of-the-box support for asynchronous programming (`async`/`await`).
  - Auto-generates interactive API documentation (Swagger UI at `/docs`).

### Uvicorn
- **What it is**: An ASGI web server implementation for Python.
- **Role in StyleLens**: It runs the event loop, handles the incoming HTTP connections, and passes them to the FastAPI application.
- **Run Command**:
  ```bash
  python -m uvicorn api.main:app --port 8000 --reload
  ```

---

## Architecture Concepts

### ASGI (Asynchronous Server Gateway Interface)
- The modern standard for Python asynchronous web servers and applications.
- Allows handling multiple concurrent connections (such as WebSockets, slow clients, and background tasks) asynchronously without blocking execution threads.

### WSGI (Web Server Gateway Interface)
- The older, traditional synchronous Python web standard (used by Flask/Django).
- Handles requests one at a time per thread/process, which is less efficient for high-concurrency or real-time workloads.
