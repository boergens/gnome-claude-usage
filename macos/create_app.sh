#!/bin/bash
# Create a macOS .app bundle for Claude Usage menu bar

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="Claude Usage"
APP_DIR="$SCRIPT_DIR/$APP_NAME.app"

# Use venv python
PYTHON_BIN="/Users/kevin/Documents/newstart/venv/bin/python"

echo "Creating $APP_NAME.app..."

# Create app structure
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# Create Info.plist
cat > "$APP_DIR/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Claude Usage</string>
    <key>CFBundleIdentifier</key>
    <string>com.claude.usage</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>LSUIElement</key>
    <true/>
    <key>LSBackgroundOnly</key>
    <false/>
</dict>
</plist>
PLIST

# Create launcher script
cat > "$APP_DIR/Contents/MacOS/launcher" << LAUNCHER
#!/bin/bash
cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" "$SCRIPT_DIR/claude_usage_menubar.py"
LAUNCHER

chmod +x "$APP_DIR/Contents/MacOS/launcher"

echo "Created: $APP_DIR"
echo ""
echo "To run: open '$APP_DIR'"
echo "To add to Login Items: System Preferences → Users & Groups → Login Items → Add '$APP_DIR'"
