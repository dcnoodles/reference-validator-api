FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer across
# rebuilds that only change application code (api_server.py / the validator).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_server.py reference_validator-GUIDE-1.py ./

# Hugging Face Spaces (Docker SDK) expects the container to listen on 7860.
ENV PORT=7860
EXPOSE 7860

CMD ["python", "api_server.py"]
