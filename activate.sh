#!/bin/bash

# Activate the Python virtual environment
source .venv-py312/bin/activate

# Source project aliases
source .aliases

# Print success message
echo "✅ Virtual environment activated and project aliases loaded."
echo "Available aliases:"
cat .aliases | grep "^alias" | sed 's/alias /  • /' | sed 's/=.*//' 