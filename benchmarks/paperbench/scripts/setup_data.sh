#!/bin/bash
#
# Paperbench Data Setup Script
#
# This script automates the process of cloning the frontier-evals repository
# and fetching the LFS data needed for paperbench evaluation.
#
# Usage:
#   ./setup_data.sh [install_location]
#
# If install_location is not provided, defaults to ~/paperbench_data
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default installation location
DEFAULT_INSTALL_DIR="$HOME/paperbench_data"
INSTALL_DIR="${1:-$DEFAULT_INSTALL_DIR}"

echo -e "${GREEN}Paperbench Data Setup${NC}"
echo "======================================"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is not installed.${NC}"
    echo "Please install git first."
    exit 1
fi

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo -e "${RED}Error: git-lfs is not installed.${NC}"
    echo ""
    echo "Install git-lfs using one of the following:"
    echo "  - Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "  - macOS: brew install git-lfs"
    echo "  - Other: https://git-lfs.github.com/"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓${NC} git and git-lfs are installed"
echo ""

# Create installation directory
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo "Installation directory: $INSTALL_DIR"
echo ""

# Clone the repository if it doesn't exist
REPO_DIR="$INSTALL_DIR/frontier-evals"
if [ -d "$REPO_DIR" ]; then
    echo -e "${YELLOW}Repository already exists at: $REPO_DIR${NC}"
    echo "Using existing repository..."
    cd "$REPO_DIR"
else
    echo "Cloning frontier-evals repository..."
    echo "(Using --filter=blob:none to avoid downloading all LFS files)"
    git clone https://github.com/leandermaben/frontier-evals.git --filter=blob:none
    cd frontier-evals
    echo -e "${GREEN}✓${NC} Repository cloned"
fi

echo ""

# Initialize git-lfs
echo "Initializing git-lfs..."
git lfs install
echo -e "${GREEN}✓${NC} git-lfs initialized"
echo ""

# Fetch LFS data for paperbench
echo "Fetching LFS data for paperbench..."
echo "(This may take a few minutes depending on your connection)"
git lfs fetch --include "project/paperbench/data/**"
echo -e "${GREEN}✓${NC} LFS data fetched"
echo ""

# Checkout the LFS files
echo "Checking out LFS files..."
git lfs checkout project/paperbench/data
echo -e "${GREEN}✓${NC} LFS files checked out"
echo ""

# Verify the data directory
DATA_DIR="$REPO_DIR/project/paperbench/data"
if [ ! -d "$DATA_DIR/papers" ]; then
    echo -e "${RED}Error: Papers directory not found at $DATA_DIR/papers${NC}"
    echo "Something went wrong during setup."
    exit 1
fi

# Count the number of papers
PAPER_COUNT=$(ls -1 "$DATA_DIR/papers" | wc -l)
echo -e "${GREEN}✓${NC} Verified data directory"
echo -e "${GREEN}✓${NC} Found $PAPER_COUNT paper directories"
echo ""

# Show environment variable setup
echo "======================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "======================================"
echo ""
echo "To use paperbench evaluation, set the following environment variable:"
echo ""
echo -e "${YELLOW}export PAPERBENCH_DATA_DIR=\"$DATA_DIR\"${NC}"
echo ""
echo "To make this permanent, add it to your shell configuration:"
echo "  - bash: echo 'export PAPERBENCH_DATA_DIR=\"$DATA_DIR\"' >> ~/.bashrc"
echo "  - zsh:  echo 'export PAPERBENCH_DATA_DIR=\"$DATA_DIR\"' >> ~/.zshrc"
echo ""
echo "Verify the setup by running:"
echo "  echo \$PAPERBENCH_DATA_DIR"
echo "  ls \$PAPERBENCH_DATA_DIR/papers/"
echo ""
