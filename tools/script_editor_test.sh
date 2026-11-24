#!/usr/bin/env bash
# Python scripts in the editor (src/ui/ScriptRunner.cpp): welder_ui --run-script runs each one
# the way Scripts > Run Script File does, then --quit-after-script undoes it and reports.
#
#   1. a script that builds nodes: they appear, and one undo takes the whole run back
#   2. tests/python_api_test.py, the whole API, inside the editor
#   3. a script that fails: its traceback reaches the output and the exit code is kept
#   4. a script that hangs: Stop ends it, and the editor keeps going
#   5. callbacks: an init.py stand-in (--callbacks-script) whose welder.on() handlers see a
#      script's changes, a save, and a node's Python button; each callback is one undo step
#   6. welder.on() from the Script Editor says where callbacks belong
#   7. a screenshot of the Script Editor (with WELDER_SHOT=<png>)
#
#   tools/script_editor_test.sh [build dir]       (needs xvfb-run without a display)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/build}"
UI="$BUILD/welder_ui"
[ -x "$UI" ] || { echo "no $UI (build first)"; exit 1; }
WORK="$(mktemp -d -t welder-scripts-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
RUN=(env HOME="$WORK/home" timeout 240)
if [ -z "${DISPLAY:-}" ]; then RUN+=(xvfb-run -a -s "-screen 0 1600x1000x24"); fi
fail=0
check() {  # name, output, pattern
  if grep -q -- "$3" <<< "$2"; then echo "ok    $1"; else echo "FAIL  $1"; echo "$2" | tail -30; fail=1; fi
}

cat > "$WORK/build.py" <<'PY'
import time, welder
comp = welder.comp()
check = comp.create("CheckerBoard", name="Check")
grade = comp.create("Grade", inputs=[check], gain=[2, 2, 2, 1])
t = time.time()
for i in range(300):
    grade["mix"] = i / 300
print("300 knob sets in %.2f s" % (time.time() - t))
grade["mix"].set_key(1, 0)
grade["mix"].set_key(10, 1)
welder.select(grade)
welder.view(grade)
welder.set_frame(5)
print("selected", [n.name for n in welder.selected()], "viewed", welder.viewed().name, "frame", welder.frame())
PY
out=$("${RUN[@]}" "$UI" --run-script "$WORK/build.py" --quit-after-script 2>&1 || true)
check "nodes made, one undo step" "$out" "SCRIPT exit=0 nodes=3 undo=1 after_undo_nodes=1"
check "editor commands" "$out" "selected \['Grade1'\] viewed Grade1 frame 5"

out=$("${RUN[@]}" "$UI" --run-script "$ROOT/tests/python_api_test.py" --quit-after-script 2>&1 || true)
check "the whole API in the editor" "$out" "all scripting API checks passed"

printf 'import welder\nprint("before")\nwelder.comp()["NoSuchNode"]\n' > "$WORK/fails.py"
out=$("${RUN[@]}" "$UI" --run-script "$WORK/fails.py" --quit-after-script 2>&1 || true)
check "a failing script's traceback" "$out" "KeyError: 'no node named NoSuchNode'"
check "its exit code" "$out" "SCRIPT exit=1"

printf 'import time\nprint("waiting", flush=True)\ntime.sleep(600)\n' > "$WORK/hangs.py"
out=$(WELDER_TEST_SCRIPT_STOP=2 "${RUN[@]}" "$UI" --run-script "$WORK/hangs.py" --quit-after-script 2>&1 || true)
check "Stop ends a script that hangs" "$out" "hangs.py: stopped after"

cat > "$WORK/callbacks.py" <<'PY'
import welder

@welder.on("knob_changed", type="Grade", knob="mix")
def mix(ev):
    print("CB mix", ev.node.name, ev.value, flush=True)
    ev.node["label"] = "mix %s" % ev.value

@welder.on("node_created")
def made(ev):
    print("CB created", ev.node.name, ev.type, flush=True)

@welder.on("comp_saved")
def saved(ev):
    print("CB saved", ev.path.endswith("cb.weld"), flush=True)

@welder.on("node_created", type="Grade")
def broken(ev):
    raise RuntimeError("a handler that fails")

welder.run_callbacks()
PY
cat > "$WORK/sets.py" <<PY
import welder
c = welder.comp()
g = c.create("Grade", name="G")
g.add_button("Half", "import welder\nn = welder.this_node()\nprint('BUTTON on', n.name)\nn['mix'] = 0.5\n")
print("buttons", [b["label"] for b in g.buttons()])
c.save("$WORK/cb.weld")
PY
out=$(WELDER_TEST_PRESS_BUTTON=G:Half WELDER_TEST_DUMP_PARAMS=1 "${RUN[@]}" "$UI" --callbacks-script "$WORK/callbacks.py" \
      --run-script "$WORK/sets.py" --quit-after-script 2>&1 || true)
check "a node created: its callback" "$out" "CB created G Grade"
check "a save: its callback" "$out" "CB saved True"
check "a failing callback's traceback" "$out" "RuntimeError: a handler that fails"
check "a Python button runs with this_node()" "$out" "BUTTON on G"
check "the button's knob change: its callback" "$out" "CB mix G 0.5"
check "undo takes back the callback, not the button" "$out" 'SCRIPT params G {"mix":0.5}'

printf 'import welder\nwelder.on("comp_saved", print)\n' > "$WORK/on.py"
out=$("${RUN[@]}" "$UI" --run-script "$WORK/on.py" --quit-after-script 2>&1 || true)
check "callbacks belong in init.py" "$out" "callbacks run from init.py"

if [ -n "${WELDER_SHOT:-}" ]; then
  "${RUN[@]}" "$UI" --run-script "$WORK/build.py" --screenshot "$WELDER_SHOT" > /dev/null 2>&1 || true
  [ -f "$WELDER_SHOT" ] && echo "ok    screenshot $WELDER_SHOT"
fi
[ $fail = 0 ] && echo "all script editor checks passed"
exit $fail
