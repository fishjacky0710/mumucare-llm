FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./hf_cache/intfloat-multi-e5-base /app/intfloat-multi-e5-base

COPY . .

ENTRYPOINT ["sh", "-c", "python main.py"]


