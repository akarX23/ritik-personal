#!/bin/bash

export PIP_CACHE_DIR="/opt/bitnami/spark"
pip install sentence_transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"