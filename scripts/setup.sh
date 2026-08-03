#!/usr/bin/env bash
# Developer setup script for Purce
# Run: bash scripts/setup.sh

set -e

echo "=== Purce Developer Setup ==="

# Check Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

# Install package in editable mode with dev dependencies
echo "Installing purce in editable mode..."
pip install -e ".[dev]" 2>&1 | tail -3

# Verify installation
echo "Verifying installation..."
python -c "from purce import __version__; print(f'Purce v{__version__} installed')"

# Run tests
echo "Running test suite..."
python -m pytest tests/ -q --tb=line 2>&1 | tail -1

# Run quick fuzz check
echo "Running fuzz verification..."
python -c "
from purce.verifier.fuzzer import DifferentialFuzzer
r = DifferentialFuzzer(seed=42).fuzz_all(iterations=20)
fails = [k for k,v in r.items() if not v.all_passed]
print(f'Fuzz: {len(r)} ops, {len(fails)} failures')
"

echo ""
echo "Setup complete. Quick start:"
echo "  purce compile <source.py> -o <output_dir>"
echo "  purce verify <output_dir>"
echo "  python -m pytest tests/"
echo "  python -m tests.verification_agent"
