#!/bin/bash
# Fetch Claude Code usage information using tmux to interact with claude CLI
#
# Usage:
#   ./fetch_usage.sh          # Normal mode - outputs key=value pairs
#   ./fetch_usage.sh --debug  # Debug mode - shows raw tmux output

DEBUG_MODE=false
if [[ "$1" == "--debug" || "$1" == "-d" ]]; then
    DEBUG_MODE=true
fi

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

# Capture the pane content (usage info)
OUTPUT=$(tmux capture-pane -t "$SESSION_NAME" -p)

# Navigate left twice to get account info
tmux send-keys -t "$SESSION_NAME" Left
sleep 0.5
tmux send-keys -t "$SESSION_NAME" Left
sleep 1

# Capture the account info
ACCOUNT_OUTPUT=$(tmux capture-pane -t "$SESSION_NAME" -p)

# Debug mode: show raw output and exit
if $DEBUG_MODE; then
    echo "=== RAW USAGE OUTPUT ==="
    echo "$OUTPUT"
    echo ""
    echo "=== RAW ACCOUNT OUTPUT ==="
    echo "$ACCOUNT_OUTPUT"
    echo ""
    # Clean up and exit
    tmux send-keys -t "$SESSION_NAME" Escape
    sleep 0.5
    tmux send-keys -t "$SESSION_NAME" '/exit' Enter
    sleep 1
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null
    exit 0
fi

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
# Pass both outputs with a separator
(echo "$OUTPUT"; echo "===ACCOUNT_INFO==="; echo "$ACCOUNT_OUTPUT") | $PYTHON_BIN -c '
import sys
import re
import os

# Add script directory to path for importing neural_process
script_dir = os.environ.get("SCRIPT_DIR", ".")
sys.path.insert(0, script_dir)

full_text = sys.stdin.read()

# Split usage and account info
parts = full_text.split("===ACCOUNT_INFO===")
text = parts[0] if parts else full_text
account_text = parts[1] if len(parts) > 1 else ""

# Find account/email from account info
account_match = re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", account_text)
account_email = account_match.group(1) if account_match else None

# Also try to find plan type (Max, Pro, etc.)
plan_match = re.search(r"(Max|Pro|Team|Enterprise|Free)", account_text, re.IGNORECASE)
plan_type = plan_match.group(1) if plan_match else None

# Find session reset time - format: "Resets 10am" or "Resets 9:59am" after "Current session"
session_reset_match = re.search(r"Current session.*?Resets\s+(\d{1,2}(?::\d{2})?[ap]m)", text, re.DOTALL | re.IGNORECASE)
session_reset = session_reset_match.group(1) if session_reset_match else None

# Find weekly reset time - format: "Resets Jan 24 at 8pm" or "Resets Jan 24 at 7:59pm" after "Current week (all models)"
weekly_reset_match = re.search(r"Current week \(all models\).*?Resets\s+(\w+\s+\d+\s+at\s+\d{1,2}(?::\d{2})?[ap]m)", text, re.DOTALL | re.IGNORECASE)
weekly_reset = weekly_reset_match.group(1) if weekly_reset_match else None

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

# Output account info
if account_email:
    print(f"ACCOUNT_EMAIL={account_email}")
if plan_type:
    print(f"PLAN_TYPE={plan_type}")

# Output reset times
if session_reset:
    print(f"SESSION_RESETS={session_reset}")
if weekly_reset:
    print(f"WEEKLY_RESETS={weekly_reset}")

# Try to get predictions from neural process
time_remaining = None
time_remaining_str = "??"
confidence = None

if session_used is not None:
    try:
        from neural_process import UsagePredictor
        predictor = UsagePredictor()

        # Record observation for future training (with reset times if available)
        if weekly_used is not None:
            predictor.record_observation(
                session_used, weekly_used,
                session_resets=session_reset,
                weekly_resets=weekly_reset
            )

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

        # Check if usage will exhaust before reset
        if session_reset and time_remaining is not None:
            try:
                from datetime import datetime, timedelta
                import re as re2

                # Parse reset time like "10am", "9:59am", or "10:30pm"
                reset_match = re2.match(r"(\d{1,2})(?::(\d{2}))?([ap]m)", session_reset.lower())
                if reset_match:
                    hour = int(reset_match.group(1))
                    minute = int(reset_match.group(2)) if reset_match.group(2) else 0
                    ampm = reset_match.group(3)

                    if ampm == "pm" and hour != 12:
                        hour += 12
                    elif ampm == "am" and hour == 12:
                        hour = 0

                    now = datetime.now()
                    reset_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

                    # If reset time is in the past, it means tomorrow
                    if reset_time <= now:
                        reset_time += timedelta(days=1)

                    hours_until_reset = (reset_time - now).total_seconds() / 3600
                    print(f"HOURS_UNTIL_RESET={hours_until_reset:.2f}")

                    # Compare: will we exhaust before reset?
                    if time_remaining < hours_until_reset:
                        print("EXHAUSTS_BEFORE_RESET=true")  # Will run out before quota refreshes
                    else:
                        print("EXHAUSTS_BEFORE_RESET=false")  # Quota refreshes before we run out
            except Exception:
                pass

    except Exception as e:
        # Neural process not available or failed
        print(f"PREDICTION_ERROR={str(e)[:50]}")

# Human readable output
if account_email:
    print(f"Account: {account_email}")
print(f"Session: {session_remaining}% remaining", end="")
if time_remaining_str != "??":
    print(f" (~{time_remaining_str})")
else:
    print()
print(f"Weekly: {weekly_remaining}% remaining")
'
