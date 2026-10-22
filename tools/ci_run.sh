#!/usr/bin/env bash
# Run a CI step, keeping its output; when it fails, the last lines become a GitHub annotation
# (readable through the API without downloading the log).
#   tools/ci_run.sh <title> <command...>
title=$1; shift
log=$(mktemp)
set -o pipefail
"$@" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
if [ "$rc" != 0 ]; then
  # the step's tail, plus the tail of any build log the dependency scripts left
  tailtxt=$(tail -n 40 "$log")
  for f in $(ls -t ../deps/*/build/*.log ../deps/*/build/make.log 2>/dev/null | head -2); do
    tailtxt="$tailtxt
--- $f
$(tail -n 25 "$f")"
  done
  msg=$(printf '%s' "$tailtxt" | sed 's/%/%25/g' | awk 'BEGIN{ORS="%0A"} {print}')
  echo "::error title=$title failed::$msg"
fi
exit "$rc"
