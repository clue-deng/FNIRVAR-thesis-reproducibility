#!/usr/bin/env python3
from pathlib import Path
import hashlib, sys
root=Path(__file__).resolve().parents[1]
bad=[]; n=0
for line in (root/"SHA256SUMS.txt").read_text().splitlines():
    if not line.strip(): continue
    expected,rel=line.split("  ",1); n+=1; p=root/rel
    if not p.is_file(): bad.append((rel,"missing")); continue
    found=hashlib.sha256(p.read_bytes()).hexdigest()
    if found!=expected: bad.append((rel,"hash mismatch"))
if bad:
    for rel,reason in bad: print("FAIL",rel,reason)
    sys.exit(1)
print("OK:",n,"files match SHA256SUMS.txt")
