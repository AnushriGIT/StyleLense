/* ── MyStoreHelp Chat UI ─────────────────────────────────────────────────────── */

const MAX_FILE_SIZE = 5 * 1024 * 1024;   // 5 MB
const $ = id => document.getElementById(id);

let attachedFile       = null;
let isRecording        = false;
let recognition        = null;
let _selectedAgentPath = "product_search";

// ── Preference card state ─────────────────────────────────────────────────────
let _pendingPrefs  = null;   // live mutable copy of the profile being edited
let _originalPrefs = null;   // snapshot at card render time (for diff)
let _prefUserQuery = null;   // original search query, replayed on confirm

// ── Auth state ────────────────────────────────────────────────────────────────
// Token is stored in localStorage so the user stays logged in across page reloads.

let _authToken   = null;    // raw JWT string
let _authUser    = null;    // decoded payload: { sub (username), name, role, customer_id, store_id, permissions }
let _sessionId   = null;    // chat history session UUID returned by /auth/login

const ROLE_LABELS = {
  customer:         "Customer",
  store_associate:  "Store Associate",
  corporate_user:   "Corporate User",
};

// Decode the JWT payload (middle section) — base64 only, no signature check.
// Signature verification happens server-side on every API call.
function decodeJwtPayload(token) {
  try {
    const base64 = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    return JSON.parse(atob(base64));
  } catch { return null; }
}

// ── Auth tab switcher ─────────────────────────────────────────────────────────
function showTab(tab) {
  $("form-login").style.display    = tab === "login"    ? "" : "none";
  $("form-register").style.display = tab === "register" ? "" : "none";
  $("tab-login").classList.toggle("active",    tab === "login");
  $("tab-register").classList.toggle("active", tab === "register");
  $("login-error").textContent    = "";
  $("register-error").textContent = "";
}

// Show/hide extra fields based on selected role
function updateRegisterFields() {
  const role = $("reg-role").value;
  $("field-customer-id").style.display = role === "customer"        ? "" : "none";
  $("field-store-id").style.display    = role === "store_associate" ? "" : "none";
}

// ── Login handler ─────────────────────────────────────────────────────────────
async function handleLogin(e) {
  e.preventDefault();
  const btn = $("login-btn");
  btn.disabled = true;
  btn.textContent = "Signing in…";
  $("login-error").textContent = "";

  try {
    const res  = await fetch("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: $("login-username").value.trim(),
        password: $("login-password").value,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Login failed.");
    onAuthSuccess(data.access_token, data.session_id || "");
  } catch (err) {
    $("login-error").textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Sign In";
  }
}

// ── Register handler ──────────────────────────────────────────────────────────
async function handleRegister(e) {
  e.preventDefault();
  const btn = $("register-btn");
  btn.disabled = true;
  btn.textContent = "Creating account…";
  $("register-error").textContent = "";

  const role    = $("reg-role").value;
  const ageRaw  = $("reg-age").value.trim();
  const body = {
    username:    $("reg-username").value.trim(),
    password:    $("reg-password").value,
    name:        $("reg-name").value.trim(),
    role,
    email:       $("reg-email").value.trim() || null,
    customer_id: role === "customer"        ? ($("reg-customer-id").value.trim() || null) : null,
    store_id:    role === "store_associate" ? ($("reg-store-id").value.trim()    || null) : null,
    gender:      $("reg-gender").value || null,
    age:         ageRaw ? parseInt(ageRaw, 10) : null,
  };

  try {
    const res  = await fetch("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Registration failed.");

    // Auto-login after successful registration
    const loginRes  = await fetch("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: body.username, password: body.password }),
    });
    const loginData = await loginRes.json();
    if (!loginRes.ok) throw new Error("Registered — To login please sign in manually.");
    onAuthSuccess(loginData.access_token, loginData.session_id || "");
  } catch (err) {
    $("register-error").textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Create Account";
  }
}

// ── Called after successful login or register ─────────────────────────────────
function onAuthSuccess(token, sessionId = "") {
  _authToken = token;
  _authUser  = decodeJwtPayload(token);
  _sessionId = sessionId;
  localStorage.setItem("auth_token", token);
  if (sessionId) localStorage.setItem("chat_session_id", sessionId);

  // Update header
  const role  = _authUser?.role || "";
  const name  = _authUser?.name || _authUser?.sub || "User";
  $("user-name").textContent  = name;
  $("role-badge").textContent = ROLE_LABELS[role] || role;
  $("role-badge").className   = `role-badge role-${role.replace("_", "-")}`;
  $("user-info").style.display = "";

  // Hide login overlay, show chat
  $("auth-overlay").style.display = "none";

  // Filter agent pills to this role
  initAgentSelector();

  // Show personalised welcome if not already shown
  if (chatMessages.children.length === 0) {
    addBotText(
      `Welcome back, <strong>${escapeHtml(name)}</strong>! 🛍️<br>` +
      `You're signed in as <em>${ROLE_LABELS[role] || role}</em>. ` +
      "Attach a clothing photo to find similar items, type a question, or tap 🎤 to search by voice."
    );
  }
}

// ── Agent path selector ───────────────────────────────────────────────────────
function initAgentSelector() {
  const role  = _authUser?.role || "anonymous";
  const pills = document.querySelectorAll(".agent-pill");

  pills.forEach(pill => {
    const allowedRoles = pill.dataset.roles.split(",");
    if (allowedRoles.includes(role)) {
      pill.style.display = "";
    } else {
      pill.style.display = "none";
      // If the hidden pill was selected, reset to product_search
      if (pill.dataset.path === _selectedAgentPath) {
        _selectedAgentPath = "product_search";
      }
    }
  });

  // Activate the right pill and update placeholder
  _setActivePill(_selectedAgentPath);
}

function _setActivePill(path) {
  _selectedAgentPath = path;
  document.querySelectorAll(".agent-pill").forEach(p => {
    p.classList.toggle("active", p.dataset.path === path);
    if (p.dataset.path === path) {
      chatInput.placeholder = p.dataset.placeholder;
    }
  });
}

// Wire pill clicks — single-select, skip if already active
document.querySelectorAll(".agent-pill").forEach(pill => {
  pill.addEventListener("click", () => {
    if (pill.classList.contains("active")) return;   // already selected, no-op
    _setActivePill(pill.dataset.path);
  });
});

// ── Logout ────────────────────────────────────────────────────────────────────
function handleLogout() {
  if (_sessionId) {
    fetch("/auth/logout", {
      method: "POST",
      headers: { "Content-Type": "application/json", ..._authHeaders() },
      body: JSON.stringify({ session_id: _sessionId }),
    }).catch(() => {});
  }
  _authToken  = null;
  _authUser   = null;
  _sessionId  = null;
  localStorage.removeItem("auth_token");
  localStorage.removeItem("chat_session_id");
  $("user-info").style.display    = "none";
  $("auth-overlay").style.display = "";
  chatMessages.innerHTML = "";
  $("login-username").value = "";
  $("login-password").value = "";
  showTab("login");
}

// ── Auth init — runs on page load ─────────────────────────────────────────────
(function initAuth() {
  const savedToken   = localStorage.getItem("auth_token");
  const savedSession = localStorage.getItem("chat_session_id") || "";
  if (savedToken) {
    const payload = decodeJwtPayload(savedToken);
    if (payload && payload.exp * 1000 > Date.now()) {
      onAuthSuccess(savedToken, savedSession);   // restore previous session on page reload
      return;
    }
    localStorage.removeItem("auth_token");
    localStorage.removeItem("chat_session_id");
  }
  $("auth-overlay").style.display = "";
})();

// ── Auth header helper ────────────────────────────────────────────────────────
function _authHeaders() {
  return _authToken ? { "Authorization": `Bearer ${_authToken}` } : {};
}

// ── DOM refs ──────────────────────────────────────────────────────────────────
const chatMessages  = $("chat-messages");
const chatInput     = $("chat-input");
const sendBtn       = $("send-btn");
const micBtn        = $("mic-btn");
const micIconIdle   = $("mic-icon-idle");
const micIconStop   = $("mic-icon-stop");
const fileInput     = $("chat-file-input");
const attachPreview = $("attach-preview");
const attachThumb   = $("attach-thumb");
const attachClear   = $("attach-clear");
const errorBanner   = $("error-banner");

// ── Health check ──────────────────────────────────────────────────────────────
fetch("/health")
  .then(r => r.json())
  .then(d => {
    const rc = $("chip-resnet");
    const cc = $("chip-clip");
    // Both image search and text search now use CLIP — show unified status
    if (d.clip_gallery?.ready) { rc.textContent = "CLIP Image · ready"; rc.classList.add("ready"); }
    else                       { rc.textContent = "CLIP Image · not ready"; rc.classList.add("error"); }
    if (d.clip_model?.ready)   { cc.textContent = "CLIP Text · ready";  cc.classList.add("ready"); }
    else                       { cc.textContent = "CLIP Text · not ready"; cc.classList.add("error"); }
  })
  .catch(() => {
    $("chip-resnet").textContent = "server unreachable";
    $("chip-resnet").classList.add("error");
  });

// Welcome message is shown by onAuthSuccess after login.

// ── Image attach ──────────────────────────────────────────────────────────────
fileInput.addEventListener("change", e => {
  const file = e.target.files[0];
  if (file) setAttachedFile(file);
  fileInput.value = "";
});

attachClear.addEventListener("click", clearAttachedFile);

function setAttachedFile(file) {
  hideError();
  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (!allowed.includes(file.type)) {
    showError("Only JPG, PNG and WebP images are supported.");
    return;
  }
  if (file.size > MAX_FILE_SIZE) {
    showError(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max 5 MB.`);
    return;
  }
  attachedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    attachThumb.src = e.target.result;
    attachPreview.style.display = "flex";
  };
  reader.readAsDataURL(file);
}

function clearAttachedFile() {
  attachedFile = null;
  attachThumb.src = "";
  attachPreview.style.display = "none";
}

// ── Voice input ───────────────────────────────────────────────────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (!SpeechRecognition) {
  micBtn.disabled = true;
  micBtn.title = "Voice search requires Chrome or Edge";
} else {
  recognition = new SpeechRecognition();
  recognition.lang           = "en-US";
  recognition.interimResults = true;
  recognition.continuous     = false;

  recognition.onstart = () => {
    isRecording = true;
    micBtn.classList.add("listening");
    micIconIdle.style.display = "none";
    micIconStop.style.display = "";
    chatInput.placeholder = "Listening…";
    chatInput.value = "";
    hideError();
  };

  recognition.onresult = e => {
    const transcript = Array.from(e.results)
      .map(r => r[0].transcript)
      .join("");
    chatInput.value = transcript;
  };

  recognition.onerror = e => {
    stopRecording();
    const msgs = {
      "no-speech":     "No speech detected — please try again.",
      "audio-capture": "Microphone not accessible.",
      "not-allowed":   "Microphone permission denied.",
      "network":       "Network error during voice recognition.",
    };
    showError(msgs[e.error] || `Voice error: ${e.error}`);
  };

  recognition.onend = () => {
    stopRecording();
    // If transcript was captured, focus input so user can review / send
    if (chatInput.value.trim()) chatInput.focus();
  };

  micBtn.addEventListener("click", () => {
    if (isRecording) {
      recognition.stop();
    } else {
      try { recognition.start(); }
      catch (e) { /* already running */ }
    }
  });
}

function stopRecording() {
  isRecording = false;
  micBtn.classList.remove("listening");
  micIconIdle.style.display = "";
  micIconStop.style.display = "none";
  chatInput.placeholder = "Ask about a product or attach an image to search…";
}

// ── Send ──────────────────────────────────────────────────────────────────────
sendBtn.addEventListener("click", handleSend);
chatInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
});

function handleSend() {
  const text = chatInput.value.trim();
  const file = attachedFile;

  const textlessPaths = ["auto_return", "trend_forecasting", "visual_inventory_auditor"];
  if (!text && !file && !textlessPaths.includes(_selectedAgentPath)) {
    showError("Type a message, attach an image, or use the mic.");
    return;
  }

  hideError();
  chatInput.value = "";

  if (file) {
    addUserMessage(text, file);
    clearAttachedFile();
    runImageSearch(file, text);
  } else {
    const displayText = text || `[${_selectedAgentPath.replace(/_/g, " ")}]`;
    addUserMessage(displayText, null);
    runTextSearch(text);
  }
}

// ── Text-only search → /agent/find-in-store ───────────────────────────────────
function runTextSearch(text) {
  const typingId = addBotTyping();
  sendBtn.disabled = true;

  const form = new FormData();
  form.append("customer_id", _authUser?.customer_id || "");
  form.append("user_query",  text);
  form.append("agent_path",  _selectedAgentPath);
  if (_sessionId) form.append("session_id", _sessionId);

  fetch("/agent/find-in-store", { method: "POST", headers: _authHeaders(), body: form })
    .then(r => r.json().then(data => ({ ok: r.ok, data })))
    .then(({ ok, data }) => {
      removeBotTyping(typingId);
      if (!ok) throw new Error(data.detail || data.error || "Search failed.");
      addBotResults(data);
    })
    .catch(err => {
      removeBotTyping(typingId);
      addBotText("Sorry, something went wrong: " + escapeHtml(err.message));
    })
    .finally(() => { sendBtn.disabled = false; });
}

// ── Image search → /agent/find-in-store ──────────────────────────────────────
function runImageSearch(file, message) {
  const typingId = addBotTyping();
  sendBtn.disabled = true;

  const form = new FormData();
  form.append("image",       file);
  form.append("customer_id", _authUser?.customer_id || "");
  form.append("user_query",  message || "");
  form.append("agent_path",  _selectedAgentPath);
  if (_sessionId) form.append("session_id", _sessionId);

  fetch("/agent/find-in-store", { method: "POST", headers: _authHeaders(), body: form })
    .then(r => r.json().then(data => ({ ok: r.ok, data })))
    .then(({ ok, data }) => {
      removeBotTyping(typingId);
      if (!ok) throw new Error(data.detail || data.error || "Search failed.");
      addBotResults(data);
    })
    .catch(err => {
      removeBotTyping(typingId);
      addBotText("Sorry, something went wrong: " + escapeHtml(err.message));
    })
    .finally(() => { sendBtn.disabled = false; });
}


// ── Chat message builders ─────────────────────────────────────────────────────

function addUserMessage(text, imageFile) {
  const msg = document.createElement("div");
  msg.className = "msg msg-user";

  let imageHtml = "";
  if (imageFile) {
    const src = URL.createObjectURL(imageFile);
    imageHtml = `<div class="msg-image"><img src="${src}" alt="uploaded clothing" /></div>`;
  }

  const textHtml = text
    ? `<div class="msg-text">${escapeHtml(text)}</div>`
    : "";

  msg.innerHTML = `
    <div class="msg-bubble">
      ${imageHtml}
      ${textHtml}
    </div>`;
  chatMessages.appendChild(msg);
  scrollToBottom();
}

function addBotText(html) {
  const msg = document.createElement("div");
  msg.className = "msg msg-bot";
  msg.innerHTML = `
    <div class="msg-avatar">🛍️</div>
    <div class="msg-bubble">${html}</div>`;
  chatMessages.appendChild(msg);
  scrollToBottom();
}

function addBotTyping() {
  const id = "typing-" + Date.now();
  const msg = document.createElement("div");
  msg.className = "msg msg-bot";
  msg.id = id;
  msg.innerHTML = `
    <div class="msg-avatar">🛍️</div>
    <div class="typing-dots"><span></span><span></span><span></span></div>`;
  chatMessages.appendChild(msg);
  scrollToBottom();
  return id;
}

function removeBotTyping(id) {
  const el = $(id);
  if (el) el.remove();
}

// ── Feedback ──────────────────────────────────────────────────────────────────
function sendFeedback(queryId, rating, btn) {
  if (!queryId) return;
  const form = new FormData();
  form.append("query_id", queryId);
  form.append("rating",   rating);
  fetch("/agent/feedback", { method: "POST", body: form })
    .then(() => {
      // Mark the clicked button, grey out the other
      const container = btn.closest(".feedback-row");
      container.querySelectorAll("button").forEach(b => {
        b.disabled = true;
        b.style.opacity = b === btn ? "1" : "0.3";
      });
      btn.textContent = rating === "good" ? "👍 Thanks!" : "👎 Noted";
    })
    .catch(() => {});
}

function feedbackHtml(queryId) {
  if (!queryId) return "";
  return `
    <div class="feedback-row">
      <span class="feedback-label">Were these results helpful?</span>
      <button class="feedback-btn" onclick="sendFeedback('${queryId}', 'good', this)">👍</button>
      <button class="feedback-btn" onclick="sendFeedback('${queryId}', 'bad', this)">👎</button>
    </div>`;
}

// ── Product detail modal ──────────────────────────────────────────────────────

function openProductModal(encodedPath, imgSrc) {
  const modal   = document.getElementById("product-modal");
  const imgEl   = document.getElementById("modal-img");
  const detailEl= document.getElementById("modal-detail");
  const storeIn = document.getElementById("modal-store-input");

  imgEl.src        = decodeURIComponent(imgSrc);
  detailEl.innerHTML = `<div class="modal-loading">Loading availability…</div>`;
  storeIn.value    = "";
  modal.dataset.imagePath = encodedPath;
  modal.style.display = "flex";
  document.body.style.overflow = "hidden";

  _fetchProductDetail(encodedPath, "");
}

function closeProductModal() {
  document.getElementById("product-modal").style.display = "none";
  document.body.style.overflow = "";
}

function checkDifferentStore() {
  const modal     = document.getElementById("product-modal");
  const storeId   = document.getElementById("modal-store-input").value.trim();
  const imgPath   = modal.dataset.imagePath || "";
  document.getElementById("modal-detail").innerHTML = `<div class="modal-loading">Checking ${escapeHtml(storeId) || "store"}…</div>`;
  _fetchProductDetail(imgPath, storeId);
}

function _fetchProductDetail(encodedPath, storeId) {
  const form = new FormData();
  form.append("image_path", decodeURIComponent(encodedPath));
  if (storeId) form.append("store_id", storeId);
  const headers = _authToken ? { "Authorization": `Bearer ${_authToken}` } : {};

  fetch("/agent/product-detail", { method: "POST", headers, body: form })
    .then(r => r.json())
    .then(d => _renderProductDetail(d))
    .catch(() => {
      document.getElementById("modal-detail").innerHTML =
        `<div class="modal-error">Could not load availability. Please try again.</div>`;
    });
}

function _renderProductDetail(d) {
  const stockBadge = d.in_stock
    ? `<span class="modal-badge in-stock">In Stock · ${d.stock_count} units</span>`
    : `<span class="modal-badge out-of-stock">Out of Stock</span>`;

  const sizesHtml = (d.sizes_available || []).length
    ? `<div class="modal-sizes">${(d.sizes_available || []).map(s => `<span class="modal-size-chip">${s}</span>`).join("")}</div>`
    : "";

  const locationHtml = d.in_stock
    ? `<div class="modal-location">
         <span class="modal-location-icon">📍</span>
         <div>
           <div class="modal-location-aisle">${escapeHtml(d.aisle || "Check with staff")}</div>
           ${d.bay ? `<div class="modal-location-bay">${escapeHtml(d.bay)}</div>` : ""}
         </div>
       </div>`
    : `<div class="modal-location-na">Not available at this location.</div>`;

  const priceHtml = d.price_gbp
    ? `<div class="modal-price">£${parseFloat(d.price_gbp).toFixed(2)}</div>`
    : "";

  // Associate-only fields
  const assocHtml = d.sku ? `
    <div class="modal-assoc-row"><span>SKU</span><code>${escapeHtml(d.sku)}</code></div>
    ${d.upc ? `<div class="modal-assoc-row"><span>UPC</span><code>${escapeHtml(d.upc)}</code></div>` : ""}
    ${d.vendor ? `<div class="modal-assoc-row"><span>Vendor</span><span>${escapeHtml(d.vendor)}</span></div>` : ""}
    ${typeof d.warehouse_qty === "number" ? `<div class="modal-assoc-row"><span>Warehouse</span><span>${d.warehouse_qty} units</span></div>` : ""}
  ` : "";

  document.getElementById("modal-detail").innerHTML = `
    <div class="modal-store-row">
      <span class="modal-store-label">Store:</span>
      <span class="modal-store-id">${escapeHtml(d.store_id || "unknown")}</span>
    </div>
    ${stockBadge}
    ${priceHtml}
    ${sizesHtml}
    <div class="modal-section-title">📍 In-Store Location</div>
    ${locationHtml}
    ${assocHtml ? `<div class="modal-section-title">Staff Details</div><div class="modal-assoc">${assocHtml}</div>` : ""}
  `;
}

// Close modal on backdrop click
document.addEventListener("click", e => {
  const modal = document.getElementById("product-modal");
  if (e.target === modal) closeProductModal();
});

// ── Preference card helpers ───────────────────────────────────────────────────

function _removePrefChip(btn) {
  btn.closest(".pref-chip").remove();
}

function _showAddInput(sectionId) {
  const container = document.getElementById("pref-" + sectionId);
  if (!container) return;
  const existing = container.querySelector(".pref-add-input");
  if (existing) { existing.focus(); return; }
  const input = document.createElement("input");
  input.className   = "pref-add-input";
  input.placeholder = "Add…";
  input.onkeydown = e => {
    if (e.key === "Enter" && input.value.trim()) {
      _addPrefChip(sectionId, input.value.trim());
      input.remove();
    } else if (e.key === "Escape") {
      input.remove();
    }
  };
  input.onblur = () => {
    if (input.value.trim()) _addPrefChip(sectionId, input.value.trim());
    setTimeout(() => input.remove(), 150);
  };
  const addBtn = container.querySelector(".pref-add-btn");
  container.insertBefore(input, addBtn);
  input.focus();
}

function _addPrefChip(sectionId, value) {
  const container = document.getElementById("pref-" + sectionId);
  if (!container) return;
  const chip = document.createElement("span");
  chip.className = "pref-chip";
  chip.innerHTML = `<span class="pref-chip-label">${escapeHtml(value)}</span>` +
                   `<button class="pref-chip-remove" onclick="_removePrefChip(this)" title="Remove">×</button>`;
  const addBtn = container.querySelector(".pref-add-btn");
  container.insertBefore(chip, addBtn);
}

function _collectPrefFromDom() {
  if (!_pendingPrefs) return;
  const getChips = id =>
    Array.from(document.querySelectorAll("#pref-" + id + " .pref-chip-label"))
         .map(el => el.textContent.trim());
  _pendingPrefs.preferred_styles   = getChips("styles");
  _pendingPrefs.preferred_colours  = getChips("colours");
  _pendingPrefs.avoided_materials  = getChips("avoid");
  _pendingPrefs.brands             = getChips("brands");
  const bMin = document.getElementById("pref-budget-min");
  const bMax = document.getElementById("pref-budget-max");
  if (bMin) _pendingPrefs.budget_min = parseInt(bMin.value)  || null;
  if (bMax) _pendingPrefs.budget_max = parseInt(bMax.value)  || null;
  const occ  = document.getElementById("pref-occasion");
  const sTop = document.getElementById("pref-size-top");
  if (occ)  _pendingPrefs.occasion = occ.value  || null;
  if (sTop) _pendingPrefs.size_top = sTop.value || null;
}

function _computePrefDiff(original, current) {
  const updates = {};
  for (const f of ["preferred_styles", "preferred_colours", "avoided_materials", "brands"]) {
    if (JSON.stringify(original[f] || []) !== JSON.stringify(current[f] || []))
      updates[f] = current[f];
  }
  for (const f of ["budget_min", "budget_max", "occasion", "size_top"]) {
    if ((original[f] || null) !== (current[f] || null)) updates[f] = current[f];
  }
  return updates;
}

function _confirmPreferences() {
  _collectPrefFromDom();
  const updates = _computePrefDiff(_originalPrefs || {}, _pendingPrefs || {});
  const typingId = addBotTyping();
  sendBtn.disabled = true;
  const form = new FormData();
  form.append("user_query",            _prefUserQuery || "");
  form.append("customer_id",           _authUser?.customer_id || "");
  form.append("agent_path",            "personal_stylist");
  form.append("preferences_confirmed", "true");
  if (Object.keys(updates).length > 0)
    form.append("preference_updates", JSON.stringify(updates));
  if (_sessionId) form.append("session_id", _sessionId);
  fetch("/agent/find-in-store", { method: "POST", headers: _authHeaders(), body: form })
    .then(r => r.json().then(data => ({ ok: r.ok, data })))
    .then(({ ok, data }) => {
      removeBotTyping(typingId);
      if (!ok) throw new Error(data.detail || "Search failed.");
      addBotResults(data);
    })
    .catch(err => {
      removeBotTyping(typingId);
      addBotText("Sorry, something went wrong: " + escapeHtml(err.message));
    })
    .finally(() => { sendBtn.disabled = false; });
}

function _buildPrefCardHtml(profile) {
  const chips = (values, sectionId) => {
    const chipsHtml = (values || []).map(v =>
      `<span class="pref-chip">` +
      `<span class="pref-chip-label">${escapeHtml(v)}</span>` +
      `<button class="pref-chip-remove" onclick="_removePrefChip(this)" title="Remove">×</button>` +
      `</span>`
    ).join("");
    return `<div class="pref-chips" id="pref-${sectionId}">${chipsHtml}` +
           `<button class="pref-add-btn" onclick="_showAddInput('${sectionId}')" title="Add">+</button></div>`;
  };

  const sizeOpts = ["XS","S","M","L","XL","XXL"].map(s =>
    `<option value="${s}"${profile.size_top === s ? " selected" : ""}>${s}</option>`).join("");
  const occOpts  = ["","weekend","work","evening","gym","casual","formal"].map(o =>
    `<option value="${o}"${profile.occasion === o ? " selected" : ""}>${o || "— any occasion —"}</option>`).join("");

  const loyaltyHtml = profile.loyalty_tier
    ? `<div class="pref-loyalty">✨ ${profile.loyalty_tier.charAt(0).toUpperCase() + profile.loyalty_tier.slice(1)} member · ${profile.loyalty_points || 0} pts</div>`
    : "";
  const recentHtml = (profile.recent_searches || []).length
    ? `<div class="pref-recent">Recent searches: ${profile.recent_searches.map(s => `<em>${escapeHtml(s)}</em>`).join(", ")}</div>`
    : "";

  return `<div class="pref-card">
    ${loyaltyHtml}${recentHtml}
    <div class="pref-row"><div class="pref-label">Style</div>${chips(profile.preferred_styles, "styles")}</div>
    <div class="pref-row"><div class="pref-label">Colour</div>${chips(profile.preferred_colours, "colours")}</div>
    <div class="pref-row"><div class="pref-label">Avoid</div>${chips(profile.avoided_materials, "avoid")}</div>
    <div class="pref-row"><div class="pref-label">Brands</div>${chips(profile.brands, "brands")}</div>
    <div class="pref-row">
      <div class="pref-label">Budget</div>
      <div class="pref-budget">
        £<input id="pref-budget-min" type="number" class="pref-budget-input" min="0" max="9999"
               value="${profile.budget_min || ""}" placeholder="min" />
        —
        £<input id="pref-budget-max" type="number" class="pref-budget-input" min="0" max="9999"
               value="${profile.budget_max || ""}" placeholder="max" />
      </div>
    </div>
    <div class="pref-row">
      <div class="pref-label">Size</div>
      <select id="pref-size-top" class="pref-select">${sizeOpts}</select>
    </div>
    <div class="pref-row">
      <div class="pref-label">Occasion</div>
      <select id="pref-occasion" class="pref-select">${occOpts}</select>
    </div>
    <div class="pref-actions">
      <button class="pref-confirm-btn" onclick="_confirmPreferences()">✓ Confirm &amp; Search</button>
    </div>
  </div>`;
}

// Monotonic counter so each styling message gets globally unique accordion IDs
let _msgCounter = 0;

function addBotResults(data) {
  const msg          = document.createElement("div");
  msg.className      = "msg msg-bot";
  const responseText = data.response_text || "";
  const aiTextHtml   = responseText
    ? `<div class="msg-bubble ai-response">${escapeHtml(responseText)}</div>`
    : "";

  // ── Preference confirmation card ──────────────────────────────────────────
  if (data.type === "preference_confirmation") {
    const profile = data.profile || {};
    _originalPrefs = JSON.parse(JSON.stringify(profile));
    _pendingPrefs  = JSON.parse(JSON.stringify(profile));
    _prefUserQuery = data.user_query || "";

    msg.innerHTML = `
      <div class="msg-avatar">🛍️</div>
      <div class="msg-bot-body">
        ${aiTextHtml}
        ${_buildPrefCardHtml(profile)}
      </div>`;
    chatMessages.appendChild(msg);
    scrollToBottom();
    return;
  }

  // ── Styling intent: accordion of 5 archetypes ─────────────────────────────
  if (data.intent === "styling" && Array.isArray(data.styling_options) && data.styling_options.length) {
    const msgId   = "style-" + (++_msgCounter);   // unique per message
    const options = data.styling_options;
    const tags    = data.attribute_tags || [];
    const tagsHtml = tags.length
      ? `<div class="tags-row">${tags.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join("")}</div>`
      : "";

    const noImgSrc = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='130'%3E%3Crect fill='%23f5f5f7' width='100' height='130'/%3E%3C/svg%3E";

    const accordionHtml = options.map((opt, i) => {
      const products = opt.products || [];
      const gridHtml = products.length
        ? products.map(p => {
            const imgSrc = p.image_data
              ? `data:image/jpeg;base64,${p.image_data}`
              : noImgSrc;
            const sim  = Math.min(100, Math.max(0, p.similarity || 0));
            const path = encodeURIComponent(p.image_path || "");
            return `
              <div class="result-card" onclick="openProductModal('${path}','${escapeHtml(imgSrc)}')" title="Click for availability & location">
                <img src="${imgSrc}" alt="Styling item" loading="lazy"
                     onerror="this.src='${noImgSrc}'" />
                <div class="card-info">
                  <div class="card-sim">${sim}% match</div>
                  <div class="sim-bar"><div class="sim-fill" style="width:${sim}%"></div></div>
                </div>
                <div class="card-tap-hint">Tap for details</div>
              </div>`;
          }).join("")
        : `<div class="accordion-empty">No matching items found for this look.</div>`;

      return `
        <div class="accordion-item" id="${msgId}-${i}">
          <button class="accordion-toggle" onclick="toggleAccordion('${msgId}', ${i})">
            <span class="accordion-name">${escapeHtml(opt.name)}</span>
            <span class="accordion-chevron">▾</span>
          </button>
          <div class="accordion-body">
            <p class="accordion-subtitle">${escapeHtml(opt.description)}</p>
            <div class="accordion-grid">${gridHtml}</div>
          </div>
        </div>`;
    }).join("");

    const totalProducts = options.reduce((s, o) => s + (o.products || []).length, 0);

    msg.innerHTML = `
      <div class="msg-avatar">🛍️</div>
      <div class="msg-bot-body">
        ${aiTextHtml}
        <div class="styling-card">
          <div class="results-card-header">
            <span class="confidence-badge medium">✨ Styling</span>
            <span class="result-count">${options.length} looks · ${totalProducts} items</span>
          </div>
          ${tagsHtml}
          <div class="accordion">${accordionHtml}</div>
        </div>
        ${feedbackHtml(data.query_id)}
      </div>`;

    chatMessages.appendChild(msg);
    scrollToBottom();
    return;
  }

  // ── Text-only error or redirect: just show AI bubble, no results card ────────
  if (!data.confidence && (!data.results || data.results.length === 0) && responseText) {
    msg.innerHTML = `
      <div class="msg-avatar">🛍️</div>
      <div class="msg-bot-body">${aiTextHtml}</div>`;
    chatMessages.appendChild(msg);
    scrollToBottom();
    return;
  }

  // ── Search / availability: results grid ───────────────────────────────────
  const conf    = data.confidence || "low";
  const score   = data.confidence_score || 0;
  const tags    = data.attribute_tags || [];
  const results = data.results || [];
  const pivot   = data.pivot_used;
  const pivotQ  = data.pivot_query;
  const count   = data.result_count || results.length;

  const confLabel = conf.charAt(0).toUpperCase() + conf.slice(1);
  const confDot   = { high: "🟢", medium: "🟡", low: "🔴" }[conf] || "";

  const tagsHtml = tags.length
    ? `<div class="tags-row">${tags.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join("")}</div>`
    : "";

  const pivotHtml = pivot
    ? `<span class="pivot-badge">↩ text pivot: "${escapeHtml(pivotQ || "")}"</span>`
    : "";

  const noImgSrc = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='130'%3E%3Crect fill='%23f5f5f7' width='100' height='130'/%3E%3C/svg%3E";

  // ── Item ID reference card (shown when user searched by catalogue ID) ────────
  let matchedItemHtml = "";
  if (data.matched_item && data.matched_item.item_id) {
    const mi     = data.matched_item;
    const miSrc  = mi.image_data ? `data:image/jpeg;base64,${mi.image_data}` : noImgSrc;
    const miPath = encodeURIComponent(mi.image_path || "");
    matchedItemHtml = `
      <div class="matched-item-row">
        <span class="matched-item-label">Searched for:</span>
        <div class="matched-item-card" onclick="openProductModal('${miPath}','${escapeHtml(miSrc)}')" title="Click for availability & location">
          <img src="${miSrc}" alt="${escapeHtml(mi.item_id)}" onerror="this.src='${noImgSrc}'" />
          <div class="matched-item-id">${escapeHtml(mi.item_id)}</div>
          <div class="card-tap-hint">Tap for details</div>
        </div>
      </div>`;
  }

  const cardsHtml = results.map(r => {
    const sim    = Math.min(100, Math.max(0, r.similarity || 0));
    const imgSrc = r.image_data
      ? `data:image/jpeg;base64,${r.image_data}`
      : noImgSrc;
    const path   = encodeURIComponent(r.image_path || "");
    return `
      <div class="result-card" onclick="openProductModal('${path}','${escapeHtml(imgSrc)}')" title="Click for availability & location">
        <img src="${imgSrc}" alt="Match #${r.rank}" loading="lazy"
             onerror="this.src='${noImgSrc}'" />
        <div class="card-info">
          <div class="card-rank">Match #${r.rank}</div>
          <div class="card-sim">${sim}% similar</div>
          <div class="sim-bar"><div class="sim-fill" style="width:${sim}%"></div></div>
        </div>
        <div class="card-tap-hint">Tap for details</div>
      </div>`;
  }).join("");

  msg.innerHTML = `
    <div class="msg-avatar">🛍️</div>
    <div class="msg-bot-body">
      ${aiTextHtml}
      <div class="results-card">
        <div class="results-card-header">
          <span class="confidence-badge ${conf}">${confDot} ${confLabel} · ${score}%</span>
          ${pivotHtml}
          <span class="result-count">${count} result${count !== 1 ? "s" : ""}</span>
        </div>
        ${tagsHtml}
        ${matchedItemHtml}
        <div class="results-grid">${cardsHtml}</div>
      </div>
      ${feedbackHtml(data.query_id)}
    </div>`;

  chatMessages.appendChild(msg);
  scrollToBottom();
}

// ── Accordion toggle ──────────────────────────────────────────────────────────
function toggleAccordion(msgId, index) {
  const item = document.getElementById(msgId + "-" + index);
  if (item) item.classList.toggle("open");
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showError(msg) {
  errorBanner.textContent = "⚠️  " + msg;
  errorBanner.style.display = "block";
}
function hideError() {
  errorBanner.style.display = "none";
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}


// ── Chat History Panel ─────────────────────────────────────────────────────────

function openHistoryPanel() {
  $("history-overlay").style.display = "flex";
  loadHistorySessions();
}

function closeHistoryPanel(event) {
  if (!event || event.target === $("history-overlay")) {
    $("history-overlay").style.display = "none";
  }
}

function loadHistorySessions() {
  const body = $("history-body");
  body.innerHTML = `<div class="history-loading">Loading…</div>`;

  fetch("/history/sessions?limit=20", { headers: _authHeaders() })
    .then(r => r.json())
    .then(data => {
      const sessions = data.sessions || [];
      if (!sessions.length) {
        body.innerHTML = `<div class="history-empty">No past sessions found.</div>`;
        return;
      }
      body.innerHTML = sessions.map(s => {
        const date   = s.started_at ? new Date(s.started_at).toLocaleString() : "Unknown";
        const turns  = s.turn_count || 0;
        return `
          <div class="history-session" onclick="loadSessionMessages('${escapeHtml(s.session_id)}', this)">
            <div class="history-session-date">${escapeHtml(date)}</div>
            <div class="history-session-meta">${turns} turn${turns !== 1 ? "s" : ""}</div>
          </div>`;
      }).join("");
    })
    .catch(() => {
      body.innerHTML = `<div class="history-empty">Could not load history.</div>`;
    });
}

function loadSessionMessages(sessionId, clickedEl) {
  // Highlight selected session
  document.querySelectorAll(".history-session").forEach(el => el.classList.remove("active"));
  clickedEl.classList.add("active");

  const body = $("history-body");
  // Keep session list, append messages below
  const existing = body.querySelectorAll(".history-messages");
  existing.forEach(el => el.remove());

  const msgBox = document.createElement("div");
  msgBox.className = "history-messages";
  msgBox.innerHTML = `<div class="history-loading">Loading messages…</div>`;
  body.appendChild(msgBox);

  fetch(`/history/sessions/${encodeURIComponent(sessionId)}`, { headers: _authHeaders() })
    .then(r => r.json())
    .then(data => {
      const msgs = data.messages || [];
      if (!msgs.length) {
        msgBox.innerHTML = `<div class="history-empty">No messages in this session.</div>`;
        return;
      }
      msgBox.innerHTML = msgs.map(m => {
        const isHuman = m.role === "human";
        const time    = m.created_at ? new Date(m.created_at).toLocaleTimeString() : "";
        return `
          <div class="hmsg hmsg-${isHuman ? "human" : "bot"}">
            <span class="hmsg-role">${isHuman ? "You" : "Assistant"}</span>
            <span class="hmsg-time">${escapeHtml(time)}</span>
            <div class="hmsg-content">${escapeHtml(m.content)}</div>
          </div>`;
      }).join("");
    })
    .catch(() => {
      msgBox.innerHTML = `<div class="history-empty">Could not load messages.</div>`;
    });
}
