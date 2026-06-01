FROM python:3.12-slim

WORKDIR /app

# Copiar requirements primero para aprovechar cache de Docker
COPY requirements.txt /app/requirements.txt

# Instalar dependencias del sistema y Python
RUN apt-get update && apt-get install -y curl && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copiar todo el código de la aplicación
COPY . /app/

EXPOSE 8800
