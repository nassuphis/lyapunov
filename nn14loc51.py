#!/usr/bin/env python
import sys
from math import isclose

def parse_target_args(arg_str):
    # arg_str like "1,-.16,0.5,li2(x)"
    parts = [p.strip() for p in arg_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected 4 arguments, got {len(parts)} in {arg_str!r}")
    a = float(parts[0])
    b = float(parts[1])
    c = float(parts[2])
    d = parts[3]  # keep as string (e.g. "li2(x)")
    return a, b, c, d

def extract_meixpol_args(line):
    """
    Find meixpol(....) in the line and return its 4 arguments as strings.
    Assumes exactly one meixpol(...) per line.
    """
    key = "meixpol("
    start = line.find(key)
    if start == -1:
        return None

    # position of first '(' after 'meixpol'
    i = start + len(key) - 1  # index of '('
    depth = 0
    end = None
    for j in range(i, len(line)):
        ch = line[j]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                end = j
                break

    if end is None:
        return None

    arg_str = line[i + 1:end]  # between outermost parentheses
    # since inner functions are single-arg like li2(x), simple split is OK
    parts = [p.strip() for p in arg_str.split(",")]
    if len(parts) != 4:
        return None
    return parts

def nearly_equal(a, b, rtol=0.1, atol=1e-9):
    # generous tolerance so -.16 matches -0.1666666666...
    return isclose(a, b, rel_tol=rtol, abs_tol=atol)

def main():
    if len(sys.argv) != 2:
        print("Usage: python get_line.py \"1,-.16,0.5,li2(x)\"")
        sys.exit(1)

    filename = "nn14loc51.spec"
    target_str = sys.argv[1]

    t1, t2, t3, t4 = parse_target_args(target_str)

    with open(filename, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            parts = extract_meixpol_args(line)
            if parts is None:
                continue

            try:
                a1 = float(parts[0])
                a2 = float(parts[1])
                a3 = float(parts[2])
            except ValueError:
                continue  # not numeric in first 3 args, skip

            a4 = parts[3]

            if (
                nearly_equal(a1, t1)
                and nearly_equal(a2, t2)
                and nearly_equal(a3, t3)
                and a4 == t4
            ):
                print(f"{lineno} : {line}")
                # if you want *all* matches, remove this break
                break

if __name__ == "__main__":
    main()