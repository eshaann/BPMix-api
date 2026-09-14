# 🎧 BPMix API

The backend REST API for **[BPMix](https://bpmixdj.com)**-an intelligent audio feature extraction and harmonic track-sequencing engine built with Flask, Librosa, and Mutagen.

## Live Endpoint

* **Production Base URL:** `https://api.bpmixdj.com`
* **Web Client:** [https://bpmixdj.com](https://bpmixdj.com)
* **Frontend Repository:** [eshaann/BPMix-frontend](https://github.com/eshaann/BPMix-frontend)

---

## Features & Capabilities

* **Audio Analysis (`librosa`):** Extracts tempo (BPM) via beat-tracking algorithms and determines musical key using Constant-Q Chromagrams (`chroma_cqt`).
* **Metadata Extraction (`mutagen`):** Parses ID3 `APIC` tags to extract embedded album artwork and convert it into Base64 data URLs for client-side rendering.
* **Harmonic Track Ordering:** Implements **Camelot Wheel** harmonic mixing logic to map musical keys and calculate optimal transition sequences across uploaded tracks while minimizing tempo shifts.
* **CORS & Proxy Ready:** Pre-configured with `flask-cors` headers for cross-origin requests from Cloudflare Pages.

---

## API Reference

### 1. Health Check
* **`GET /health`**
* **Response `200 OK`:**
  ```json
  {
    "status": "ok"
  }
