"""Compile the existing document and prepare page images for visual review."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

import pdfplumber
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEX = ROOT / "docs/mobile_ev_probing_report_v2.tex"
PDF = TEX.with_suffix(".pdf")
RESULTS = HERE / "results_v7"
XELATEX = Path(r"D:\apps\texlive\2026\bin\windows\xelatex.exe")
POPPLER = Path(r"C:\Users\23761\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    began = datetime.now(timezone.utc).isoformat()
    for n in (1, 2, 3):
        run = subprocess.run([str(XELATEX), "-no-shell-escape", "-interaction=nonstopmode",
                              "-halt-on-error", TEX.name], cwd=TEX.parent, capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=180)
        (RESULTS / f"latex_final_pass_{n}.txt").write_text(run.stdout + run.stderr, encoding="utf-8")
        if run.returncode:
            raise RuntimeError(f"XeLaTeX pass {n} failed; see archived log")
    source = TEX.read_text(encoding="utf-8")
    declared = re.findall(r"\\bibitem\{([^}]+)\}", source)
    used = {k.strip() for v in re.findall(r"\\cite\{([^}]+)\}", source) for k in v.split(",")}
    labels = re.findall(r"\\label\{([^}]+)\}", source)
    log = TEX.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
    bad_log = [line for line in log.splitlines() if re.search(
        r"undefined|Rerun to get|Overfull|Missing character|multiply defined", line, re.I)]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS / f"pdf_review_{stamp}"
    out.mkdir()
    pages, edge_issues = [], []
    with pdfplumber.open(PDF) as doc:
        for n, page in enumerate(doc.pages, 1):
            text = page.extract_text() or ""
            pages.append(text)
            outside = [c for c in page.chars if c.get("text", "").strip() and
                       (c["x0"] < 1 or c["x1"] > page.width - 1 or c["top"] < 1 or c["bottom"] > page.height - 1)]
            if outside:
                edge_issues.append({"page": n, "count": len(outside)})
    combined = "\n\n".join(f"[PAGE {n}]\n{v}" for n, v in enumerate(pages, 1))
    (out / "extracted_text.txt").write_text(combined, encoding="utf-8")
    normalized = re.sub(r"\s+", "", combined)
    required = ["486", "历史相容域", "5.625", "7.500", ".000618", "严格增益的充要条件", "当前没有核验实际参考误差"]
    checks = {v: v in normalized for v in required}
    rendering = subprocess.run([str(POPPLER), "-png", "-r", "90", str(PDF), str(out / "page")],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    if rendering.returncode:
        raise RuntimeError(rendering.stderr)
    images = sorted(out.glob("page-*.png"))
    if len(images) != len(pages):
        raise RuntimeError("render count differs from PDF page count")
    contacts = []
    for start in range(0, len(images), 8):
        sheet = Image.new("RGB", (1440, 1060), "#dddddd")
        draw = ImageDraw.Draw(sheet)
        for k, path in enumerate(images[start:start + 8]):
            with Image.open(path) as im:
                thumbnail = ImageOps.contain(im.convert("RGB"), (348, 495))
            x, y = (k % 4) * 360 + 6, (k // 4) * 530 + 28
            draw.text((x, y - 22), f"PAGE {start + k + 1}", fill="black")
            sheet.paste(thumbnail, (x, y))
        target = out / f"contact-{start // 8 + 1:02d}.png"
        sheet.save(target)
        contacts.append(str(target))
    result = dict(started_at=began, completed_at=datetime.now(timezone.utc).isoformat(),
        tex=str(TEX), pdf=str(PDF), tex_sha256=sha(TEX), pdf_sha256=sha(PDF),
        compiler="local XeLaTeX, three passes, no shell escape",
        builtin_compiler="unavailable: Windows sandbox setup required",
        pages=len(pages), all_pages_rendered=len(images) == len(pages),
        content_checks=checks, log_issues=bad_log, edge_issues=edge_issues,
        bibliography_entries=len(declared), undefined_citations=sorted(used - set(declared)),
        duplicate_labels=len(labels) != len(set(labels)),
        duplicate_bibliography=len(declared) != len(set(declared)),
        section_start_pages=[n for n, page in enumerate(pages, 1) if "算例：先证明机制" in re.sub(r"\s+", "", page)],
        visual_review_completed=False, review_directory=str(out), contact_sheets=contacts)
    (RESULTS / "report_pdf_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if bad_log or edge_issues or used - set(declared) or not all(checks.values()):
        raise RuntimeError("PDF verification checks require attention")


if __name__ == "__main__":
    main()
