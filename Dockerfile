# ---- build llama.cpp (linux) ----
FROM ubuntu:22.04 AS build

RUN apt-get update && apt-get install -y \
    build-essential cmake git pkg-config \
    curl libcurl4-openssl-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone https://github.com/ggerganov/llama.cpp . \
 && cmake -S . -B build \
      -DLLAMA_SERVER=ON \
      -DLLAMA_METAL=OFF \
      -DLLAMA_CURL=ON \
      -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j 2

# ---- runtime ----
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libcurl4 \
    libstdc++6 \
    libgcc-s1 \
    libgomp1 \
    libatomic1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# llama-server
COPY --from=build /src/build/bin/llama-server /app/llama-server

# copy shared libs produced by build (if any)
# NOTE: destination MUST end with /
COPY --from=build /src/build/bin/*.so* /usr/local/lib/

RUN chmod +x /app/llama-server \
 && ldconfig

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PORT=8080
EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]