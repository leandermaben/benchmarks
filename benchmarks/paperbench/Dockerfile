# Paperbench Agent Environment (Base)
# Based on paperbench Dockerfile.base with OpenHands SDK integration

FROM ubuntu:24.04

# Environment variables for paperbench structure
ENV WORKSPACE_BASE=/home
ENV SUBMISSION_DIR=/home/submission
ENV LOGS_DIR=/home/logs
ENV CODE_DIR=/home/code
ENV AGENT_DIR=/home/agent
ENV CONDA_ENV_NAME=agent
ENV REQUIREMENTS=/home/agent/requirements.txt
ENV PYTHON_VERSION=3.12

RUN mkdir -p ${LOGS_DIR} ${AGENT_DIR} ${CODE_DIR} ${SUBMISSION_DIR}
RUN mkdir -p ${WORKSPACE_BASE}/paper
RUN mkdir ${WORKSPACE_BASE}/.vscode

# Avoid interactive dialog from apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install basic packages + git-lfs
RUN apt-get update && apt-get install -y \
    curl wget git git-lfs vim nano unzip zip p7zip-full python3 python3-pip \
    python3-venv python3-dev python-is-python3 build-essential openssh-server \
    tmux gettext sudo ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

RUN apt update && apt install -y jupyter

# Install Docker
RUN curl -fsSL https://get.docker.com -o /tmp/get-docker.sh && \
    chmod 700 /tmp/get-docker.sh && /tmp/get-docker.sh

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py313_25.5.1-0-Linux-x86_64.sh \
    -O /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && /opt/conda/bin/conda init

# Create conda environment
RUN /opt/conda/bin/conda tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/r && \
    /opt/conda/bin/conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
ENV PATH="/opt/conda/bin:${PATH}"

ENV DEBIAN_FRONTEND=

WORKDIR /home

# Initialize submission directory as git repo
RUN cd ${SUBMISSION_DIR} && git init && mkdir -p .git/hooks

# Set up git config
RUN git config --global user.email "agent@example.com" && \
    git config --global user.name "agent"

# Install OpenHands SDK components (new SDK version)
RUN pip install --no-cache-dir \
    "git+https://github.com/OpenHands/agent-sdk.git@main#subdirectory=openhands-sdk" \
    "git+https://github.com/OpenHands/agent-sdk.git@main#subdirectory=openhands-tools" \
    "git+https://github.com/OpenHands/agent-sdk.git@main#subdirectory=openhands-agent-server" \
    "git+https://github.com/OpenHands/agent-sdk.git@main#subdirectory=openhands-workspace"

# Install paperbench package
RUN pip install --no-cache-dir \
    "git+https://github.com/leandermaben/frontier-evals.git#subdirectory=project/paperbench"

# Install additional dependencies
RUN pip install --no-cache-dir \
    datasets \
    huggingface-hub \
    pandas \
    Pillow \
    openai \
    anthropic \
    jinja2

# Install Playwright for browser support
RUN pip install --no-cache-dir playwright
RUN playwright install chromium --with-deps

# For grading on the computer
RUN mkdir -p /submission /output

WORKDIR /home
EXPOSE 8000

# CRITICAL: ENTRYPOINT gets the command, CMD provides default args
# When docker run passes args, they replace CMD but ENTRYPOINT stays
ENTRYPOINT ["python", "-m", "openhands.agent_server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
