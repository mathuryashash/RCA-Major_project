"""Structural sanity check for the IEEE LaTeX source.

Not a substitute for compiling it -- no TeX distribution is installed on the
development machine, so the .tex in this directory has never been run through
pdflatex. This catches the errors that are cheap to catch without one:
unbalanced environments and braces, references with no label, citations with
no bibitem.
"""

import re
import sys
from collections import Counter
from pathlib import Path

TEX = Path(__file__).with_name("localrca_ieee.tex")


def main() -> int:
    source = TEX.read_text(encoding="utf-8")
    # Drop comments so a literal % in prose does not confuse the counts.
    body = re.sub(r"(?<!\\)%.*", "", source)

    opened = Counter(re.findall(r"\\begin\{([A-Za-z]+\*?)\}", body))
    closed = Counter(re.findall(r"\\end\{([A-Za-z]+\*?)\}", body))
    unbalanced = {
        name: (opened[name], closed[name])
        for name in set(opened) | set(closed)
        if opened[name] != closed[name]
    }

    braces_open, braces_close = body.count("{"), body.count("}")
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", body))
    cite_groups = re.findall(r"\\cite\{([^}]+)\}", body)
    cites = {c.strip() for group in cite_groups for c in group.split(",")}
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", body))

    problems = []
    if unbalanced:
        problems.append(f"unbalanced environments: {unbalanced}")
    if braces_open != braces_close:
        problems.append(f"braces {braces_open} open vs {braces_close} close")
    if refs - labels:
        problems.append(f"references with no label: {sorted(refs - labels)}")
    if cites - bibitems:
        problems.append(f"citations with no bibitem: {sorted(cites - bibitems)}")

    print(f"environments : {sum(opened.values())} balanced"
          if not unbalanced else "environments : UNBALANCED")
    print(f"braces       : {braces_open} / {braces_close}")
    print(f"labels/refs  : {len(labels)} labels, {len(refs)} refs")
    print(f"citations    : {len(cites)} cited, {len(bibitems)} bibitems"
          f"{'' if not (bibitems - cites) else f', uncited {sorted(bibitems - cites)}'}")
    print()
    for env in ("figure", "table", "equation", "tikzpicture", "axis"):
        print(f"  {env:13} {opened[env]}")
    print(f"  {'sections':13} {len(re.findall(r'\\section\{', body))}")

    print()
    if problems:
        for problem in problems:
            print(f"PROBLEM: {problem}")
        return 1
    print("no structural problems found (still unverified by a TeX run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
