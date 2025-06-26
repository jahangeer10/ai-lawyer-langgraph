# Setup Guide

This project provides a simple LangGraph implementation of an AI lawyer agent. The backend is built with FastAPI and the frontend uses Next.js.

## Prerequisites

- Python 3.10+
- Node.js (for the `ui` package)

## Installation

1. Create a virtual environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

3. Run the FastAPI server:

```bash
uvicorn src.api.main:app --reload
```

4. (Optional) Install frontend dependencies and start the Next.js dev server:

```bash
cd ui
npm install
npm run dev
```

The API will be available at `http://localhost:8000` and the frontend at `http://localhost:3000` by default.
