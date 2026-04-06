FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

# Install system dependencies needed by scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Pre-create the workspace so we can install deps before the volume mount
WORKDIR /workspaces/state
