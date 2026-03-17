# FROM python:3.11-slim

# WORKDIR /app

# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# CMD ["python", "train.py"]
# 1. Lightweight base image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements first (layer caching)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy project files (INCLUDING mnist.csv)
COPY . .

# 6. Default command (aligned with train.py)
CMD ["python", "train.py", "--csv-path", "mnist.csv"]