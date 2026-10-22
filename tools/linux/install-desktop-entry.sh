#!/usr/bin/env bash
# Adds Strata to your desktop's application menu (for you only; nothing outside your home folder).
# Run it again after moving the folder. Remove: rm ~/.local/share/applications/strata.desktop
set -e
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/strata.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Strata
Comment=Node-based compositor
Exec="$HERE/Strata" %f
Path=$HERE
Terminal=false
Categories=Graphics;Video;AudioVideo;
MimeType=application/x-strata;
DESK
[ -f "$HERE/bin/strata.png" ] && echo "Icon=$HERE/bin/strata.png" >> "$HOME/.local/share/applications/strata.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
echo "Strata is in your applications menu."
