#!/usr/bin/env bash
# Adds Welder to your desktop's application menu (for you only; nothing outside your home folder).
# Run it again after moving the folder. Remove: rm ~/.local/share/applications/welder.desktop
set -e
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/welder.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Welder
Comment=Node-based compositor
Exec="$HERE/Welder" %f
Path=$HERE
Terminal=false
Categories=Graphics;Video;AudioVideo;
MimeType=application/x-welder;
DESK
[ -f "$HERE/bin/welder.png" ] && echo "Icon=$HERE/bin/welder.png" >> "$HOME/.local/share/applications/welder.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
echo "Welder is in your applications menu."
