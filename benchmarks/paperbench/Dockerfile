# Paperbench Docker Image
# Based on Ubuntu 24.04 with NVIDIA support for GPU-enabled reproduction tasks

FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    python3.12 \
    python3-pip \
    python3.12-venv \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Create a virtual environment and activate it
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install OpenHands SDK components (new SDK version)
RUN pip install --no-cache-dir \
    "git+https://github.com/OpenHands/agent-sdk.git@main#subdirectory=openhands-sdk" \
    "git+https://github.com/OpenHands/agent-sdk.git@main#subdirectory=openhands-tools" \
    "git+https://github.com/OpenHands/agent-sdk.git@main#subdirectory=openhands-agent-server" \
    "git+https://github.com/OpenHands/agent-sdk.git@main#subdirectory=openhands-workspace"

# Install paperbench dependencies
RUN pip install --no-cache-dir \
    "git+https://github.com/leandermaben/frontier-evals.git#subdirectory=project/paperbench"

# Install additional dependencies for paperbench
RUN pip install --no-cache-dir \
    datasets \
    huggingface-hub \
    pandas \
    Pillow \
    openai \
    anthropic \
    jinja2

# Set working directory
WORKDIR /workspace

# Entrypoint for the agent server
ENTRYPOINT ["python", "-m", "openhands.agent_server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
