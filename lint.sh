#!/bin/bash

# Run black and isort formatters directly
echo "Running formatters on all Python files..."

# Run black formatter
echo "Running black..."
black . --exclude ".venv-py312/|.venv/|.venv-py313/|__pycache__/"

# Run isort formatter
echo "Running isort..."
isort . --skip ".venv-py312" --skip ".venv" --skip ".venv-py313"

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "✅ All formatting completed successfully!"
else
    echo "❌ Formatting failed. Please check the errors above."
fi

exit $exit_code 