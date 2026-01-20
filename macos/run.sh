#!/bin/bash
# Run the Claude Usage menu bar app

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use venv python if available (for torch and rumps)
PYTHON_BIN="python3"
if [ -x "/Users/kevin/Documents/newstart/venv/bin/python" ]; then
    PYTHON_BIN="/Users/kevin/Documents/newstart/venv/bin/python"
fi

# Check for rumps
if ! $PYTHON_BIN -c "import rumps" 2>/dev/null; then
    echo "Installing rumps..."
    $PYTHON_BIN -m pip install rumps
fi

# Run the app
exec $PYTHON_BIN "$SCRIPT_DIR/claude_usage_menubar.py"
