FROM docker.io/library/python:3.12.4-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python train_models.py

EXPOSE 5000
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120"]