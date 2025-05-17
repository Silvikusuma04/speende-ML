FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy requirements.txt ke dalam container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh project ke dalam container
COPY . /app/

# Expose port 5000 (port default Flask)
EXPOSE 5000

# Perintah untuk menjalankan aplikasi
CMD ["python", "app.py"]