FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./hf_cache/bge-small-zh-v1.5 /app/hf_cache/bge-small-zh-v1.5

COPY . .

ENTRYPOINT ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]

