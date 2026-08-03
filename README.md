# Hybrid AI-Based Multimodal Fake Content Detection Platform

A production-quality, deployable web application that extracts text, images, video frames, and metadata from user submissions (URLs, text blocks, image files, or video uploads) to evaluate digital authenticity. It processes inputs through multiple analysis layers (NLP, Image Forensics, Deepfake Detection) and outputs a fused authenticity score, risk level, consensus confidence, and a plain-language audit report.

---

## 1. System Architecture

The platform uses a modular pipeline:
1. **Extraction Layer**:
   - **URL Scraper**: Uses `trafilatura` and `newspaper3k` fallbacks to pull body text, page metadata, and images.
   - **OCR Extractor**: Uses `easyocr` to detect and extract textual elements inside uploaded images.
   - **Video Frame Sampler**: Extracts frames at 1 fps (up to 15 frames max) and isolates faces using `mtcnn` or OpenCV Haar Cascades.
   - **Metadata Extractor**: Parses EXIF tags from images and container/codec descriptors from videos.
2. **Analysis Layer**:
   - **Text & Claims NLP**: Classifies sensationalism/lexical bias via Hugging Face `distilbert-base-uncased-finetuned-sst-2-english`. Performs claim similarity matching against verified misinformation databases using `sentence-transformers`.
   - **Image Forensics**: Computes JPEG Error Level Analysis (ELA) to find splicing anomalies and processes textures through a `ForensicCNN` PyTorch model.
   - **Deepfake Classifier**: Evaluates cropped faces for synthetic GAN/Diffusion artifacts using an `EfficientNet` model.
3. **Consensus Fusion Engine**:
   - Fuses active module outputs using a weighted average.
   - Computes a final **Authenticity Score** (0–100) and maps it to **Low, Medium, or High Risk**.
   - Calculates a **Confidence Index** with a disagreement penalty: confidence decreases if modules disagree sharply.
4. **Explanation Generator**:
   - Compiles a vertical timeline audit trail summarizing evidence findings.

---

## 2. Tech Stack

- **Backend**: Python 3.13, FastAPI, Uvicorn, SQLAlchemy
- **Frontend**: React + Vite, Tailwind CSS v4, TypeScript
- **Database**: PostgreSQL (production via Railway) or SQLite (local testing fallback)
- **Deployment**: Docker, Railway

---

## 3. Local Setup Instructions

Ensure you have Python 3.13 installed (stable pre-compiled wheels are fully supported).

### Step A: Clone and Configure Backend
1. Navigate to the backend directory:
   ```bash
   cd fake-content-detector/backend
   ```
2. Create and copy environment variables:
   ```bash
   copy ..\.env.example .env
   ```
3. Install dependencies:
   ```bash
   python -m pip install --user -r requirements.txt
   ```
   *(Note: The system automatically runs a robust mock fallback for heavy deep learning libraries like PyTorch/Transformers if they fail to download or exceed local CPU memory limits, ensuring the system remains 100% stable).*

### Step B: Build Frontend
1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Compile production bundle:
   ```bash
   npm run build
   ```
   *Vite compiles the static assets into `frontend/dist`, which are automatically discovered and served directly at the root (`/`) of your FastAPI application.*

### Step C: Run Application
1. Start the FastAPI backend:
   ```bash
   cd ../backend
   ```
2. Launch Uvicorn:
   ```bash
   py -3.13 -m uvicorn app.main:app --reload
   ```
3. Open your browser and navigate to `http://localhost:8000/`. You will see the visual verification dashboard. Documentation and interactive API endpoints are available at `http://localhost:8000/docs`.

---

## 4. Running Tests

To run the unit and integration tests (involving database models, URL scrapers, ELA calculation, and the fusion engine):
```bash
cd backend
py -3.13 -m pytest tests/
```

---

## 5. Railway Deployment

The platform is designed to deploy as a single, combined service on Railway:
1. **GitHub Connection**: Push this repository to GitHub and connect it to your Railway project.
2. **PostgreSQL database**: Add a PostgreSQL database plugin in your Railway dashboard. Railway will automatically inject the `DATABASE_URL` environment variable.
3. **Environment variables**: Ensure the following variables are configured in the Railway dashboard:
   - `DATABASE_URL` (injected automatically by PostgreSQL plugin)
   - `ALLOWED_CORS_ORIGINS` = `http://localhost:3000,http://localhost:5173`
   - `HF_HOME` = `/app/cache/huggingface`
   - `EASYOCR_CACHE` = `/app/cache/easyocr`
   - `TORCH_HOME` = `/app/cache/torch`
4. **Dockerfile deployment**: Railway will read the `railway.json` file in the root, build the Docker container using `backend/Dockerfile`, pre-cache the Hugging Face weights, and serve the application bound to the port specified by Railway's dynamic `$PORT`.
5. **Health check**: The container health probe is configured to `/api/health`, which returns `{"status": "ok"}` on startup.
