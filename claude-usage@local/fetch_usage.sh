#!/bin/bash
# Fetch Claude Code usage information using tmux to interact with claude CLI

SESSION_NAME="claude_usage_fetch_$$"

# Kill any existing session with this name
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Start a new detached tmux session running claude
tmux new-session -d -s "$SESSION_NAME" -x 120 -y 50 'claude --dangerously-skip-permissions'

# Wait for startup
sleep 2

# Press enter (for trust dialog if shown)
tmux send-keys -t "$SESSION_NAME" Enter
sleep 2

# Press enter again (in case of second prompt)
tmux send-keys -t "$SESSION_NAME" Enter
sleep 2

# Type /usage with spacing
for char in '/' 'u' 's' 'a' 'g' 'e'; do
    tmux send-keys -t "$SESSION_NAME" -l "$char"
    sleep 0.2
done

# Press Enter
tmux send-keys -t "$SESSION_NAME" Enter

# Wait for output to render
sleep 2

# Capture the pane content
OUTPUT=$(tmux capture-pane -t "$SESSION_NAME" -p)

# Kill the session
tmux send-keys -t "$SESSION_NAME" Escape
sleep 0.5
tmux send-keys -t "$SESSION_NAME" '/exit' Enter
sleep 1
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Get script directory for neural_process.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_DIR

# Use venv python if available (for torch), otherwise system python
PYTHON_BIN="python3"
if [ -x "/Users/kevin/Documents/newstart/venv/bin/python" ]; then
    PYTHON_BIN="/Users/kevin/Documents/newstart/venv/bin/python"
fi

# Parse the output for usage percentages and get predictions
echo "$OUTPUT" | $PYTHON_BIN -c '
import sys
import re
import os

# Add script directory to path for importing neural_process
script_dir = os.environ.get("SCRIPT_DIR", ".")
sys.path.insert(0, script_dir)

text = sys.stdin.read()

# Find session usage
session_match = re.search(r"Current session.*?(\d+)%\s*used", text, re.DOTALL)
session_pct = session_match.group(1) if session_match else None

# Find weekly usage (all models)
weekly_match = re.search(r"Current week \(all models\).*?(\d+)%\s*used", text, re.DOTALL)
weekly_pct = weekly_match.group(1) if weekly_match else None

# Find weekly sonnet usage
sonnet_match = re.search(r"Current week \(Sonnet only\).*?(\d+)%\s*used", text, re.DOTALL)
sonnet_pct = sonnet_match.group(1) if sonnet_match else None

# Find extra usage
extra_match = re.search(r"Extra usage.*?(\d+)%\s*used", text, re.DOTALL)
extra_pct = extra_match.group(1) if extra_match else None

# Calculate remaining
if session_pct:
    session_used = int(session_pct)
    session_remaining = 100 - session_used
else:
    session_used = None
    session_remaining = "??"

if weekly_pct:
    weekly_used = int(weekly_pct)
    weekly_remaining = 100 - weekly_used
else:
    weekly_used = None
    weekly_remaining = "??"

# Output basic info
session_out = session_pct if session_pct else "??"
weekly_out = weekly_pct if weekly_pct else "??"
sonnet_out = sonnet_pct if sonnet_pct else "??"
print(f"SESSION_USED={session_out}")
print(f"SESSION_REMAINING={session_remaining}")
print(f"WEEKLY_USED={weekly_out}")
print(f"WEEKLY_REMAINING={weekly_remaining}")
print(f"SONNET_USED={sonnet_out}")
if extra_pct:
    print(f"EXTRA_USED={extra_pct}")

# Try to get predictions from neural process
time_remaining = None
time_remaining_str = "??"
confidence = None

if session_used is not None:
    try:
        from neural_process import UsagePredictor
        predictor = UsagePredictor()

        # Record observation for future training
        if weekly_used is not None:
            predictor.record_observation(session_used, weekly_used)

        # Get prediction
        result = predictor.predict_depletion(session_used)
        time_remaining = result["time_remaining_hours_mean"]
        confidence = result["confidence"]

        # Format time remaining
        if time_remaining <= 0:
            time_remaining_str = "<5m"  # Almost depleted
        elif time_remaining >= 1:
            time_remaining_str = f"{time_remaining:.1f}h"
        else:
            time_remaining_str = f"{int(time_remaining * 60)}m"

        print(f"TIME_REMAINING={time_remaining:.2f}")
        print(f"TIME_REMAINING_STR={time_remaining_str}")
        print(f"CONFIDENCE={confidence:.2f}")

    except Exception as e:
        # Neural process not available or failed
        print(f"PREDICTION_ERROR={str(e)[:50]}")

# Human readable output
print(f"Session: {session_remaining}% remaining", end="")
if time_remaining_str != "??":
    print(f" (~{time_remaining_str})")
else:
    print()
print(f"Weekly: {weekly_remaining}% remaining")
'
