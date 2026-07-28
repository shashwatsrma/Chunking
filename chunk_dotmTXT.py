"""
GovAgent — DOTM Directive Chunker
==================================
Reads a marked-up docx OR txt (chapters marked with #, list items
indented) and produces hierarchical, leaf-level JSON chunks with
parent ancestry metadata, ready for embedding.

Usage:
    pip install python-docx
    python3 chunk_dotm.py

Place your input files in the same folder as this script (or edit
INPUT_FILES below), then run it. Output JSON files land in
./chunks_output/, one file per chapter.

TXT MODE NOTES:
  - A list item line must be indented with a TAB or leading spaces
    for it to be recognized — this is how the script tells a list
    item apart from a heading, since plain text has no "indent
    setting" the way Word does.
  - If your export used spaces instead of tabs, check
    TXT_INDENT_MIN_SPACES below and adjust if needed.
  - This mode has NOT been tested against real files yet — run it
    on one chapter first and check the output/warnings carefully
    before trusting it the way the docx path is already trusted.
"""

import re
import json
from pathlib import Path

# ---------------------------------------------------------------------
# CONFIG — edit these if your filenames/paths differ
# ---------------------------------------------------------------------
INPUT_FILES = ["AAGYA.txt", "ANISH.txt", "ISHA.txt"]
# For txt files, just list them instead, e.g.:
# INPUT_FILES = ["AAGYA.txt", "ANISH.txt", "ISHA.txt"]

OUTPUT_DIR = Path("chunks_outputTXT")
LONG_CHUNK_WARNING_CHARS = 3000   # flag chunks bigger than this for review
TXT_INDENT_MIN_SPACES = 4          # a line needs at least this many leading spaces (or a tab) to count as indented

# ---------------------------------------------------------------------
# Devanagari numeral helpers
# ---------------------------------------------------------------------
DEVNUM = "०१२३४५६७८९"
DEVLETTER = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"


def devnum_to_int(s: str) -> int:
    """Convert a Devanagari numeral string (e.g. '१३') to a Python int."""
    return int("".join(str(DEVNUM.index(ch)) for ch in s))


# ---------------------------------------------------------------------
# Step 1 — extract paragraphs with text + indentation.
# Two readers: one for .docx, one for .txt. The rest of the script
# (classify / bundle) doesn't care which one produced the rows — both
# return the same shape: [{"text": ..., "indented": True/False}, ...]
# ---------------------------------------------------------------------
def extract_rows_docx(path: Path):
    import docx
    from docx.oxml.ns import qn

    d = docx.Document(str(path))
    rows = []
    for para in d.paragraphs:
        # Pull every w:t node under this paragraph, including ones
        # nested inside sdt content-control wrappers (Google Docs
        # export quirk) — python-docx's own .text property misses these.
        t_nodes = para._p.findall(".//" + qn("w:t"))
        text = "".join(node.text or "" for node in t_nodes).strip()
        if not text:
            continue

        pf = para.paragraph_format
        first_line = pf.first_line_indent
        left_indent = pf.left_indent
        is_indented = bool(
            (first_line is not None and first_line.twips != 0)
            or (left_indent is not None and left_indent.twips != 0)
        )
        rows.append({"text": text, "indented": is_indented})
    return rows


def extract_rows_txt(path: Path):
    """
    Reads a plain .txt export. Since plain text has no formatting,
    a line counts as 'indented' only if it starts with a TAB or at
    least TXT_INDENT_MIN_SPACES leading spaces.

    encoding='utf-8-sig' strips a leading BOM if present (we saw
    'UTF-8 with BOM' on these exports earlier — utf-8-sig handles
    that automatically, plain 'utf-8' would leave a stray character
    on the very first line and break its classification).
    """
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            leading_ws = len(raw_line) - len(raw_line.lstrip(" \t"))
            is_indented = raw_line.startswith("\t") or leading_ws >= TXT_INDENT_MIN_SPACES
            rows.append({"text": stripped, "indented": is_indented})
    return rows


def extract_rows(path: Path):
    if path.suffix.lower() == ".txt":
        return extract_rows_txt(path)
    return extract_rows_docx(path)


# ---------------------------------------------------------------------
# Step 2 — classify each line
# ---------------------------------------------------------------------
def classify(row):
    t = row["text"]

    if t.startswith("परिच्छेद"):
        return ("chapter", None, t)

    # Multi-part numbers, e.g. ११.१. or ११.१.१.  -> always a heading,
    # never a list item (list items only ever use a single bare number).
    m = re.match(rf"^([{DEVNUM}]+(?:\.[{DEVNUM}]+)+)\.\s*(.*)", t)
    if m:
        num = m.group(1)
        depth = num.count(".") + 1     # ११.१ -> depth 2, ११.१.१ -> depth 3
        return (f"heading_L{depth}", num, m.group(2))

    # Lettered heading, e.g. (क) title  -> letters are never used for
    # plain leaf list items in this document, so no marker needed.
    m = re.match(rf"^\(([{DEVLETTER}])\)\s*#?\s*(.*)", t)
    if m:
        return ("lettered_heading", m.group(1), m.group(2))

    # Bare single number, optionally marked with # -> the one
    # genuinely ambiguous case. # = heading. No # + indented = list item.
    m = re.match(rf"^([{DEVNUM}]+)\.(#?)\s*(.*)", t)
    if m:
        num, hashmark, rest = m.groups()
        if hashmark:
            return ("heading_L1", num, rest)
        elif row["indented"]:
            return ("leaf_item", num, rest)
        else:
            # Should not happen in a fully-marked file — flagged, not dropped.
            return ("UNRESOLVED_AMBIGUOUS", num, rest)

    # Parenthetical numeral leaf item, e.g. (१) text
    m = re.match(rf"^\(([{DEVNUM}]+)\)\s*(.*)", t)
    if m:
        return ("leaf_item_paren", m.group(1), m.group(2))

    return ("prose", None, t)


# ---------------------------------------------------------------------
# Step 3 — bundle classified rows into leaf chunks with parent ancestry
# ---------------------------------------------------------------------
def bundle_chunks(rows, source_file):
    chunks = []
    warnings = []
    current = None
    current_chapter = None          # int, e.g. 3
    ancestry_nums = {}               # {1: '११', 2: '११.१', 3: '११.१.१', 'letter': 'क'}

    def flush():
        nonlocal current
        if current is None:
            return
        if not (current["content"].strip() or current["title"]):
            current = None
            return
        content = current["content"].strip()
        refs = sorted(set(re.findall(r"अनुसूची-[०-९]+", content)))
        chunk = {
            "chunk_id": current["chunk_id"],
            "chapter": current_chapter,
            "clause_number": ancestry_nums.get(1),
            "subclause_number": ancestry_nums.get(2),
            "subsubclause_number": ancestry_nums.get(3),
            "lettered_subheading": current.get("letter"),
            "title": current["title"],
            "content_type": current["ctype"],
            "content": content,
            "referenced_annexes": refs,
            "source_file": source_file,
            "needs_verification": False,
        }
        if len(content) > LONG_CHUNK_WARNING_CHARS:
            warnings.append(
                f"LONG CHUNK ({len(content)} chars): {chunk['chunk_id']} — {chunk['title'][:50]}"
            )
        chunks.append(chunk)
        current = None

    def chunk_id_for_heading(num):
        # num from a heading_L match is already the FULL dotted path
        # (e.g. '११.१.१'), so it alone is enough — no need to prepend
        # parent levels again.
        return f"dotm_ch{current_chapter}_{num.replace('.', '_')}"

    def chunk_id_for_letter(letter):
        deepest = None
        for d in sorted((k for k in ancestry_nums if isinstance(k, int)), reverse=True):
            deepest = ancestry_nums[d]
            break
        prefix = deepest.replace(".", "_") if deepest else ""
        return f"dotm_ch{current_chapter}_{prefix}_{letter}" if prefix else f"dotm_ch{current_chapter}_{letter}"

    for row in rows:
        kind, num, text = classify(row)

        if kind == "chapter":
            flush()
            m = re.search(rf"[{DEVNUM}]+", row["text"])
            current_chapter = devnum_to_int(m.group(0)) if m else current_chapter
            ancestry_nums = {}
            continue

        if kind == "UNRESOLVED_AMBIGUOUS":
            warnings.append(
                f"AMBIGUOUS LINE (no # and not indented) in ch.{current_chapter}: "
                f"'{num}. {text[:40]}' — check this line by hand in {source_file}"
            )
            # Best-effort: treat as a list item so nothing silently vanishes.
            if current is not None:
                current["content"] += f"{num}. {text} "
            continue

        if kind == "heading_L1":
            flush()
            ancestry_nums = {1: num}
            current = {
                "chunk_id": chunk_id_for_heading(num), "title": text,
                "content": "", "ctype": "clause", "letter": None,
            }
        elif kind and kind.startswith("heading_L"):
            depth = int(kind.split("L")[1])
            flush()
            # drop any deeper ancestry levels, then set this one
            ancestry_nums = {d: v for d, v in ancestry_nums.items() if isinstance(d, int) and d < depth}
            ancestry_nums[depth] = num
            current = {
                "chunk_id": chunk_id_for_heading(num), "title": text,
                "content": "", "ctype": "section", "letter": None,
            }
        elif kind == "lettered_heading":
            flush()
            current = {
                "chunk_id": chunk_id_for_letter(num), "title": text,
                "content": "", "ctype": "subsection", "letter": num,
            }
        elif kind in ("leaf_item", "leaf_item_paren", "prose"):
            if current is None:
                continue
            prefix = f"{num}. " if kind == "leaf_item" else ""
            current["content"] += prefix + text + " "

    flush()
    return chunks, warnings


# ---------------------------------------------------------------------
# Step 4 — run it on a file, split output by chapter
# ---------------------------------------------------------------------
def process_file(path: Path):
    print(f"\n{'='*60}\nProcessing {path.name}")
    rows = extract_rows(path)
    chunks, warnings = bundle_chunks(rows, path.name)

    by_chapter = {}
    for c in chunks:
        by_chapter.setdefault(c["chapter"], []).append(c)

    OUTPUT_DIR.mkdir(exist_ok=True)
    for chapter, chapter_chunks in by_chapter.items():
        out_path = OUTPUT_DIR / f"ch{chapter}_chunks.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chapter_chunks, f, ensure_ascii=False, indent=2)
        print(f"  Chapter {chapter}: {len(chapter_chunks)} chunks -> {out_path}")

    if warnings:
        print(f"  ⚠️  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"     - {w}")
    else:
        print("  No warnings.")

    return len(chunks), warnings


def main():
    total_chunks = 0
    total_warnings = 0
    for fname in INPUT_FILES:
        path = Path(fname)
        if not path.exists():
            print(f"SKIPPING {fname} — file not found in this folder.")
            continue
        n_chunks, warnings = process_file(path)
        total_chunks += n_chunks
        total_warnings += len(warnings)

    print(f"\n{'='*60}")
    print(f"DONE. {total_chunks} chunks written total. {total_warnings} warning(s) to review.")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
