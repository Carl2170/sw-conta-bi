# 1. Imagen base liviana
FROM python:3.10-slim

# 2. Setear directorio de trabajo
WORKDIR /app

# 3. Copiar archivos necesarios
COPY requirements.txt ./
COPY . .

# 4. Instalar dependencias
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5. Exponer el puerto por defecto de Flask
EXPOSE 5000

# 6. Ejecutar la aplicación
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]