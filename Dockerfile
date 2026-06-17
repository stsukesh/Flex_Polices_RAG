# Hugging Face Spaces Docker Image
# Runs the Flex Policies RAG Chatbot as a FastAPI web service
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the SentenceTransformer model during build
# This avoids downloading on every container start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Copy the rest of the application
COPY main.py .
COPY rag_pipeline/ ./rag_pipeline/
COPY static/ ./static/
COPY data/ ./data/

# Hugging Face Spaces uses port 7860 by default
ENV PORT=7860

# Expose the port
EXPOSE 7860

# Start the FastAPI server
CMD ["python", "main.py"]
