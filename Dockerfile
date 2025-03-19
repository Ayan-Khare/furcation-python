# Use a lightweight version of Python
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy and install dependencies with cache optimization
COPY requirements.txt .

# Install system dependencies and Python requirements
RUN apt-get update && apt-get install -y libgl1 \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only the necessary application files
COPY main.py .

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
