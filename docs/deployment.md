# Deployment Guide

This guide details instructions for deploying Cortex AI to various environments.

---

## 1. Local Run (Fast Setup)

Ensure you have installed packages listed in `requirements.txt` and created a `.env` file containing your `GEMINI_API_KEY`.

Run:
```bash
streamlit run ui/app.py
```
Access the application on `http://localhost:8501`.

---

## 2. Local Docker Container

To package and execute the application inside a standalone container:

### Build the Image
```bash
docker build -t cortex-ai:v1.0 .
```

### Run the Container
You must pass your `GEMINI_API_KEY` as an environment variable and map host directories to persist data:
```bash
docker run -d \
  -p 8501:8501 \
  -e GEMINI_API_KEY="your_api_key" \
  -v ${PWD}/chroma_db:/app/chroma_db \
  -v ${PWD}/pdfs:/app/pdfs \
  cortex-ai:v1.0
```

---

## 3. Docker Compose Orchestration (Recommended)

Docker Compose automatically handles persistent volume creations and configures dependencies:

```bash
# Start the service
docker-compose up -d --build

# Inspect running logs
docker-compose logs -f

# Stop the service
docker-compose down
```

---

## 4. Streamlit Community Cloud

Deploying directly from a GitHub repository to the public cloud:

1. Push your repository to GitHub (ensure `.env` is NOT committed!).
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New App**, select your repository, branch (e.g. `main`), and set the Main file path to `ui/app.py`.
4. Click **Advanced settings...** and paste your secrets configuration:
   ```toml
   GEMINI_API_KEY = "AIzaSy...your_gemini_key"
   ```
5. Click **Deploy**. Streamlit will automatically build the environment from `requirements.txt` and launch the app.

---

## 5. Troubleshooting Deployment Issues

### Port Conflict (`8501` already in use)
Change Streamlit's target port by running:
```bash
streamlit run ui/app.py --server.port=8502
```

### Permissions Error on `chroma_db/`
Ensure that the user running the container has write permissions on the mounted directory. In Linux/macOS:
```bash
chmod -R 777 chroma_db/
```

### Missing API Key
If you receive exceptions indicating that the model cannot connect or the API key is empty, verify that your `.env` contains:
```env
GEMINI_API_KEY=AIzaSy...
```
Or verify that `GEMINI_API_KEY` is exported on your host environment.
