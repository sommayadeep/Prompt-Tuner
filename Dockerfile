# Use the strict version requirement
FROM python:3.10-slim

# Set environment
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Metadata
LABEL maintainer="Sanjay Kumar"
LABEL description="RL Prompt Auto-Tuner Validation Container"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Mandatory requirement: Port 7860
EXPOSE 7860

# Mandatory requirement: uvicorn launch command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
