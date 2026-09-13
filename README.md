---
title: Reference Validator API
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Reference Validator API

Flask REST API backend for the Cross-Encoder–based citation integrity
verification and intelligent source recommendation framework. Serves
`/api/validate`, `/api/upload`, `/api/upload-and-validate`, and
`/api/analyze-thesis`. See `api_server.py` for the full endpoint list
(also available live at `/api/health`).

The frontend (`reference_validator_ui.html`) is hosted separately as a
static page and points its `BASE` constant at this Space's URL.
