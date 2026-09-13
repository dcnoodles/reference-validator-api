"""
desktop_app.py – Standalone desktop application entry point.
─────────────────────────────────────────────────────────────────────────────
Runs the exact same Flask backend as api_server.py, but instead of exposing
it as a web service, opens it in a native application window via pywebview.
Everything runs on the user's own machine — no hosting, no internet
connection required beyond the CrossRef/arXiv/Semantic Scholar lookups the
validator itself makes.

This is the entry point PyInstaller packages into a standalone executable.
api_server.py itself is untouched and still works as a plain web server —
this file only adds the desktop-app wrapper on top of it.
─────────────────────────────────────────────────────────────────────────────
"""

import threading
import webview

from api_server import app

PORT = 5000


def _run_flask():
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    flask_thread = threading.Thread(target=_run_flask, daemon=True)
    flask_thread.start()

    webview.create_window(
        "Reference Validator",
        f"http://127.0.0.1:{PORT}",
        width=1280,
        height=860,
        min_size=(900, 600),
    )
    webview.start()
