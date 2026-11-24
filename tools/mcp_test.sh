#!/usr/bin/env bash
# The editor's MCP server (src/ui/McpService.cpp): starts welder_ui with the server on
# (WELDER_MCP_PORT), then runs tests/mcp_test.py --editor through `welder mcp`, the stdio bridge an
# AI client launches: it builds a comp in the open editor, reads pixels and previews, undoes, and
# checks the HTTP server's token, Host and Origin checks.
#
#   tools/mcp_test.sh [build dir]       (needs xvfb-run without a display)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/build}"
UI="$BUILD/welder_ui"
CLI="$BUILD/welder"
[ -x "$UI" ] && [ -x "$CLI" ] || { echo "no $UI or $CLI (build first)"; exit 1; }
WORK="$(mktemp -d -t welder-mcp-XXXXXX)"
mkdir -p "$WORK/home"
export HOME="$WORK/home"
PORT="${WELDER_MCP_TEST_PORT:-19471}"
RUN=(env WELDER_MCP_PORT="$PORT" timeout 300)
if [ -z "${DISPLAY:-}" ]; then RUN+=(xvfb-run -a -s "-screen 0 1600x1000x24"); fi
"${RUN[@]}" "$UI" > "$WORK/ui.log" 2>&1 &
UIPID=$!
cleanup() { kill "$UIPID" 2>/dev/null || true; pkill -f "$UI" 2>/dev/null || true; wait "$UIPID" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT
for i in $(seq 1 120); do
  [ -f "$HOME/.config/welder/mcp-server.json" ] && break
  sleep 0.5
done
[ -f "$HOME/.config/welder/mcp-server.json" ] || { echo "FAIL  the editor's MCP server didn't start"; tail -30 "$WORK/ui.log"; exit 1; }
echo "ok    the editor's MCP server is on ($(cat "$HOME/.config/welder/mcp-server.json"))"
python3 "$ROOT/tests/mcp_test.py" "$CLI" --editor
