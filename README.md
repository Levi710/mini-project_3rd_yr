---
title: Pluto Pipeline
emoji: 🚀
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Pluto: Real Mode-Switching Extraction Pipeline

This is an AI-powered document extraction pipeline that intelligently routes, understands, and verifies document data.

## Deployment on Hugging Face Spaces

This project is configured as a Docker Space. It uses:
- **Backend**: FastAPI (Python 3.10)
- **Frontend**: Custom Vanilla JS Dashboard
- **Processing**: Mode-switching extraction logic

## Local Setup
1. `pip install -r mp1/requirements.txt`
2. Configure `.env` with your `GROQ_API_KEY`
3. Run `python mp1/main.py --serve`
