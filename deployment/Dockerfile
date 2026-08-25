# Dockerfile — NeuroSleep Deployment
# Build context: repository root
# docker build -f deployment/Dockerfile -t neurosleep .

FROM python:3.11-slim

WORKDIR /app

COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY deployment/app.py .
COPY deployment/config ./config
COPY artifacts/final ./model
COPY results/final ./results/final

ENV PYTHONPATH=/app/src

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=7860"]
