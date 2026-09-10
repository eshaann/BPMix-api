FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Add environment variable default
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT} app:app"]