"""
app.py
════════════════════════════════════════════════════════════════
ai2agi — Flask Web Application
AGI ≈ f(G_enhanced, R_enhanced, ai_LLM, DTB, CV_Adapt)

Author  : Alex Osterneck, CLA, MSCS, MSIT
Org     : Ai70000, Ltd.
Version : 1.0 — May 2026
════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import traceback
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session

# ── Path setup — works both locally and on Render ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "ai2agi-ai70000-2026")

# Server-side session (keeps SESSION dict alive across requests)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = os.path.join(BASE_DIR, ".flask_session")
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
Session(app)

# ── Lazy import agi_inference (heavy — imports torch, numpy, etc.) ────────────
_agi_module = None

def get_agi():
    global _agi_module
    if _agi_module is None:
        try:
            import agi_inference as m
            _agi_module = m
        except Exception as e:
            raise RuntimeError(f"Failed to import agi_inference: {e}")
    return _agi_module


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main chat UI."""
    return render_template("index.html")


@app.route("/api/infer", methods=["POST"])
def infer():
    """
    POST /api/infer
    Body: { "task": "user question" }
    Returns: JSON with full AGI decision dict
    """
    data = request.get_json(force=True, silent=True)
    if not data or not data.get("task", "").strip():
        return jsonify({"error": "No task provided."}), 400

    task = data["task"].strip()

    try:
        m = get_agi()

        # Restore SESSION state from Flask session if available
        if "agi_session" in session:
            try:
                saved = session["agi_session"]
                m.SESSION["topic_coverage"] = set(saved.get("topic_coverage", []))
                m.SESSION["query_times"]    = saved.get("query_times", [])
                m.SESSION["query_lengths"]  = saved.get("query_lengths", [])
                m.SESSION["hormonal"]       = saved.get("hormonal", m.SESSION["hormonal"])
                # Rebuild history (embeddings can't be JSON'd, recompute)
                for h in saved.get("history", []):
                    m.SESSION["history"].append({
                        "task":      h["task"],
                        "answer":    h["answer"],
                        "embedding": m._simple_embed(h["task"]),
                    })
            except Exception:
                pass  # If restore fails, fresh session is fine

        result = m.agi_inference(task)

        # Persist SESSION state back to Flask session
        session["agi_session"] = {
            "topic_coverage": list(m.SESSION["topic_coverage"]),
            "query_times":    m.SESSION["query_times"][-20:],   # cap at 20
            "query_lengths":  m.SESSION["query_lengths"][-20:],
            "hormonal":       m.SESSION["hormonal"],
            "history": [
                {"task": h["task"], "answer": h["answer"]}
                for h in m.SESSION["history"][-10:]              # cap at 10
            ],
        }

        # Sanitize result for JSON (remove non-serializable types)
        safe = _jsonify_result(result)
        return jsonify(safe)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] /api/infer: {e}\n{tb}")
        return jsonify({"error": str(e), "traceback": tb}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset session state — clears conversation history."""
    session.clear()
    try:
        m = get_agi()
        m.SESSION["history"]        = []
        m.SESSION["topic_coverage"] = set()
        m.SESSION["query_times"]    = []
        m.SESSION["query_lengths"]  = []
        m.SESSION["hormonal"]       = {
            "cortisol": 0.1, "dopamine": 0.5,
            "norepinephrine": 0.3, "serotonin": 0.7,
        }
    except Exception:
        pass
    return jsonify({"status": "reset"})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "system": "ai2agi", "org": "Ai70000, Ltd."})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _jsonify_result(d: dict) -> dict:
    """Recursively convert result dict to JSON-safe types."""
    import numpy as np
    import torch

    safe = {}
    for k, v in d.items():
        if isinstance(v, (np.integer, np.floating)):
            safe[k] = float(v)
        elif isinstance(v, np.ndarray):
            safe[k] = v.tolist()
        elif isinstance(v, torch.Tensor):
            safe[k] = v.tolist()
        elif isinstance(v, dict):
            safe[k] = _jsonify_result(v)
        elif isinstance(v, set):
            safe[k] = list(v)
        elif isinstance(v, float) and (v != v):  # NaN
            safe[k] = 0.0
        else:
            safe[k] = v
    return safe


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
