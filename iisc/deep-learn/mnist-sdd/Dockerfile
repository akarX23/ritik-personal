FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch in a dedicated layer (large, cache-friendly)
RUN pip install --no-cache-dir torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu

# Install remaining runtime dependencies from PyPI
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy source and tests
COPY src/ src/
COPY tests/ tests/

# Data and results volumes — mount host paths or let torchvision download data
VOLUME ["/app/data"]
VOLUME ["/app/results"]

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--device", "cpu", "--data", "/app/data", "--results", "/app/results"]
