# odyra_web worker — Railway
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py .

# Scarica in build i modelli locali (Silero VAD + turn detector):
# evita il cold start al primo job.
RUN python agent.py download-files

CMD ["python", "agent.py", "start"]
