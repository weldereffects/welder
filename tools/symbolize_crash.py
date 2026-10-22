#!/usr/bin/env python3
"""Turn a Welder crash report's module+offset lines into function names.

    tools/symbolize_crash.py build-win/welder_ui.exe crash-XXXX.txt

Uses the exe's symbol table (nm), so it needs the exact exe that crashed."""
import bisect, re, subprocess, sys

exe, report = sys.argv[1], sys.argv[2]
tool = "x86_64-w64-mingw32-" if exe.lower().endswith(".exe") else ""
base = 0x140000000
for ln in subprocess.run([tool + "objdump", "-p", exe], capture_output=True, text=True).stdout.splitlines():
    if ln.startswith("ImageBase"):
        base = int(ln.split()[1], 16)
syms = []
for ln in subprocess.run([tool + "nm", "-C", "--defined-only", exe], capture_output=True, text=True).stdout.splitlines():
    parts = ln.split(" ", 2)
    if len(parts) == 3 and parts[1].lower() in "tw":
        syms.append((int(parts[0], 16), parts[2]))
syms.sort()
addrs = [a for a, _ in syms]
name = exe.replace("\\", "/").split("/")[-1].lower()
for ln in open(report, encoding="utf-8", errors="replace"):
    m = re.search(r"([\w.-]+)\+0x([0-9a-fA-F]+)", ln)
    if m and m.group(1).lower() == name:
        a = base + int(m.group(2), 16)
        i = bisect.bisect_right(addrs, a) - 1
        if i >= 0:
            ln = ln.rstrip("\n") + f"   <- {syms[i][1]} +0x{a - syms[i][0]:x}\n"
    sys.stdout.write(ln)
