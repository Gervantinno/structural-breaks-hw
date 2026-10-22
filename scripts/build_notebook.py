"""Builds an executed Jupyter notebook from a script in "percent" format.

    python scripts/build_notebook.py notebooks/analysis.py notebooks/analysis.ipynb

Cells are marked as in Jupytext/VS Code:  `# %%` — code, `# %% [markdown]` — text.
Code runs sequentially in a single namespace; stdout and matplotlib figures are
stored as outputs, so GitHub renders the notebook with all results. (If you have
Jupyter, you can simply re-run it: Kernel → Restart & Run All.)
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_cells(text: str):
    parts = re.split(r"^# %%(.*)$", text, flags=re.M)
    cells = []
    if parts[0].strip():
        cells.append(("code", parts[0].strip("\n")))
    for header, body in zip(parts[1::2], parts[2::2]):
        body = body.strip("\n")
        if "[markdown]" in header:
            lines = [re.sub(r"^# ?", "", ln) for ln in body.splitlines()]
            cells.append(("markdown", "\n".join(lines)))
        elif body.strip():
            cells.append(("code", body))
    return cells


def to_source(text: str):
    lines = text.splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return lines


def main(src: str, dst: str):
    src_p, dst_p = Path(src).resolve(), Path(dst).resolve()
    os.chdir(src_p.parent)                    # relative paths behave as when run from notebooks/
    sys.path.insert(0, str(src_p.parent))
    cells = parse_cells(src_p.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__main__"}
    nb_cells, count = [], 0
    t0 = time.time()
    for kind, text in cells:
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "id": f"md-{len(nb_cells)}", "metadata": {}, "source": to_source(text)})
            continue
        count += 1
        buf = io.StringIO()
        outputs = []
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(text, f"<cell {count}>", "exec"), ns)
        except Exception:
            print(f"Error in cell {count}:\n{text}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        out = buf.getvalue()
        if out:
            outputs.append({"output_type": "stream", "name": "stdout", "text": to_source(out)})
        for num in plt.get_fignums():
            fig = plt.figure(num)
            png = io.BytesIO()
            fig.savefig(png, format="png", dpi=100, bbox_inches="tight")
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(png.getvalue()).decode(),
                                     "text/plain": [f"<Figure {num}>"]}})
            plt.close(fig)
        nb_cells.append({"cell_type": "code", "id": f"code-{count}", "execution_count": count, "metadata": {},
                         "outputs": outputs, "source": to_source(text)})
        print(f"  cell {count} done ({time.time() - t0:.1f} s)")
    nb = {"cells": nb_cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": sys.version.split()[0]}},
          "nbformat": 4, "nbformat_minor": 5}
    dst_p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook saved: {dst_p}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
