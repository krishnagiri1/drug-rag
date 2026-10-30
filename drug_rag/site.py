"""Export a static answer explorer (for GitHub Pages or any static host).

    python -m drug_rag export-site          # writes docs/index.html + docs/data.js

Anyone can browse every evaluation question with the generated answer from each retrieval
setup, the label excerpts it cited, and the LLM judge's faithfulness and correctness verdicts,
plus the full results report. Everything is pre-computed: no server, GPU, or API key needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import INDEX_DIR, RESULTS_DIR, SECTIONS, TESTSET_PATH
from .report import MODE_NAMES

def _inline(text: str) -> str:
    import html
    import re

    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"_([^_]+)_(?=\W|$)", r"<i>\1</i>", t)
    return t


def markdown_to_html(md: str) -> str:
    """Tiny converter for the report's markdown: headings, bullets, tables, paragraphs."""
    out, para, i = [], [], 0
    lines = md.splitlines()

    def flush():
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            flush()
        elif line.startswith("#"):
            flush()
            level = min(len(line) - len(line.lstrip("#")) + 1, 4)
            out.append(f"<h{level}>{_inline(line.lstrip('#').strip())}</h{level}>")
        elif line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            body = [r for r in rows[1:] if not all(set(c) <= set("-: ") for c in r)]
            h = "".join(f"<th>{_inline(c)}</th>" for c in rows[0])
            b = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body)
            out.append(f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>')
            continue
        elif line.lstrip().startswith("- "):
            flush()
            items = []
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                items.append(f"<li>{_inline(lines[i].lstrip()[2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        else:
            para.append(line.strip())
        i += 1
    flush()
    return "\n".join(out)



BASE_CSS = r"""
:root{--bg:#fafaf9;--panel:#fff;--text:#1c1917;--muted:#57534e;--line:#e7e5e4;--accent:#4f46e5;--good:#15803d;--bad:#b91c1c;--code:#f5f5f4}
@media (prefers-color-scheme:dark){:root{--bg:#0c0a09;--panel:#1c1917;--text:#f5f5f4;--muted:#a8a29e;--line:#292524;--accent:#a5b4fc;--good:#4ade80;--bad:#f87171;--code:#141210}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:1.7rem;margin:0 0 6px;letter-spacing:-.01em}
.lede{color:var(--muted);margin:0 0 20px;max-width:760px}
a{color:var(--accent)}
nav{display:flex;gap:4px;border-bottom:1px solid var(--line);margin-bottom:18px}
nav button{background:none;border:0;border-bottom:2px solid transparent;padding:8px 12px;font:inherit;color:var(--muted);cursor:pointer}
nav button[aria-selected=true]{color:var(--text);border-color:var(--accent);font-weight:600}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:16px}
.controls label{display:flex;flex-direction:column;gap:4px;font-size:.85rem;color:var(--muted)}
select{font:inherit;padding:7px 8px;border:1px solid var(--line);border-radius:7px;background:var(--panel);color:var(--text);min-width:220px;max-width:100%}
.nav2{display:flex;gap:6px}
.nav2 button{font:inherit;padding:7px 12px;border:1px solid var(--line);border-radius:7px;background:var(--panel);color:var(--text);cursor:pointer}
.count{color:var(--muted);font-size:.85rem;margin-left:auto}
.outcome{font-weight:700;margin:4px 0 10px}
.outcome.pass{color:var(--good)}.outcome.fail{color:var(--bad)}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-size:.8rem;color:var(--muted);font-weight:600;white-space:nowrap}
th.num{text-align:center}
td:first-child{min-width:150px}
#report h2{font-size:1.15rem;margin:26px 0 8px}#report h3{font-size:1rem;margin:20px 0 6px}
#report .tablewrap{margin:10px 0 6px}#report td,#report th{white-space:nowrap}
code{font:12.5px ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--code);padding:1px 4px;border-radius:4px}
tr:last-child td{border-bottom:0}
td.num{font-variant-numeric:tabular-nums;text-align:center}
.ok{color:var(--good);font-weight:600}.no{color:var(--bad);font-weight:600}
.why{margin:12px 0 0;color:var(--text)}
.why b{color:var(--muted)}
pre{background:var(--code);border:1px solid var(--line);border-radius:10px;padding:14px;white-space:pre-wrap;word-break:break-word;font:12.5px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;max-height:70vh;overflow:auto}
h2{font-size:1.1rem;margin:22px 0 8px}
#report table{background:var(--panel)}
#report{overflow-x:auto}
footer{color:var(--muted);font-size:.85rem;margin-top:36px}
@media (max-width:600px){.controls label{width:100%}select{min-width:0;width:100%}.count{margin-left:0}}
"""


def _load_chunk_texts(index_root: Path, chunk_size: int, wanted: set[str]) -> dict[str, dict]:
    path = Path(index_root) / f"cs{chunk_size}" / "chunks.jsonl"
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["chunk_id"] in wanted:
                out[c["chunk_id"]] = c
    return out


def build_data(results_dir: Path = RESULTS_DIR, testset_path: Path = TESTSET_PATH,
               index_root: Path = INDEX_DIR) -> dict:
    results_dir = Path(results_dir)
    testset = {}
    if Path(testset_path).exists():
        with Path(testset_path).open(encoding="utf-8") as f:
            testset = {q["qid"]: q for q in map(json.loads, f) if q}

    configs, records = [], {}
    for summ_path in sorted(results_dir.glob("answers_*.summary.json")):
        meta = json.loads(summ_path.read_text(encoding="utf-8"))
        jsonl = summ_path.with_name(summ_path.name.replace(".summary.json", ".jsonl"))
        if not jsonl.exists():
            continue
        latest = {}
        with jsonl.open(encoding="utf-8") as f:
            for r in map(json.loads, f):
                if "error" not in r:
                    latest[r["qid"]] = r
        name = f"{MODE_NAMES.get(meta['mode'], meta['mode'])} · {meta['chunk_size']}-char chunks · k={meta['k']}"
        configs.append({"key": summ_path.stem.replace(".summary", ""), "name": name, "mode": meta["mode"],
                        "chunk_size": meta["chunk_size"], "summary": meta["summary"], "generator": meta["generator"]})
        records[configs[-1]["key"]] = (meta["chunk_size"], latest)
    # Best-performing config first (the reranked one, when present)
    configs.sort(key=lambda c: c["mode"] != "hybrid_rerank")

    chunk_cache: dict[int, dict] = {}
    for cs in {cs for cs, _ in records.values()}:
        wanted = {cid for c_cs, recs in records.values() if c_cs == cs for r in recs.values() for cid in r["sources"]}
        chunk_cache[cs] = _load_chunk_texts(index_root, cs, wanted)

    qids = sorted({q for _, recs in records.values() for q in recs}, key=lambda q: (q[0], q))
    items = []
    for qid in qids:
        q = testset.get(qid, {})
        answers = {}
        for cfg in configs:
            cs, recs = records[cfg["key"]]
            r = recs.get(qid)
            if not r:
                continue
            srcs = []
            for n, cid in enumerate(r["sources"], 1):
                c = chunk_cache[cs].get(cid, {})
                body = c.get("text", "").split("\n", 1)[-1]
                srcs.append({"n": n, "drug": c.get("generic_name", ""), "section": SECTIONS.get(c.get("section", ""), c.get("section", "")),
                             "text": body[:1400], "cited": n in r.get("cited", [])})
            faith = r.get("faithfulness", {})
            corr = r.get("correctness", {})
            answers[cfg["key"]] = {
                "answer": r["answer"], "refused": r["refused"], "sources": srcs,
                "verdict": corr.get("verdict"), "reason": corr.get("reason", ""),
                "faithfulness": faith.get("faithfulness"),
                "unsupported": [c.get("claim", "") for c in faith.get("claims", []) if isinstance(c, dict) and not c.get("supported")],
                "gold_retrieved": r.get("gold_retrieved"),
            }
        items.append({"qid": qid, "kind": q.get("kind", ""), "question": q.get("question") or next(
            (recs[qid]["question"] for _, recs in records.values() if qid in recs), ""),
            "reference": q.get("reference_answer", ""), "drug": q.get("drug", ""),
            "section": SECTIONS.get(q.get("gold_section", ""), q.get("gold_section", "")), "answers": answers})

    report = results_dir / "report.md"
    return {"configs": configs, "items": items,
            "report": markdown_to_html(report.read_text(encoding="utf-8")) if report.exists() else ""}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drug Label QA Explorer</title>
<meta name="description" content="Retrieval-augmented question answering over FDA drug labels: browse every evaluation question, cited answer, and judge verdict.">
<style>
__CSS__
.badge{display:inline-block;font-size:.78rem;font-weight:600;padding:2px 8px;border-radius:999px;border:1px solid var(--line);margin-right:6px}
.b-correct{color:var(--good);border-color:currentColor}.b-incorrect{color:var(--bad);border-color:currentColor}
.b-partial{color:#b45309;border-color:currentColor}.b-refused{color:var(--muted)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.card h3{margin:0 0 8px;font-size:.95rem}
.answer{font-size:1rem;margin:6px 0 10px}
.meta{color:var(--muted);font-size:.85rem}
.src{border-top:1px solid var(--line);padding:10px 0 4px}
.src:first-child{border-top:0}
.src .head{font-size:.85rem;font-weight:600}
.src .head .cited{color:var(--good)}
.src p{margin:4px 0 0;font-size:.88rem;color:var(--text);white-space:pre-wrap}
.unsup{color:var(--bad);font-size:.9rem;margin:6px 0 0}
.disclaimer{font-size:.85rem;color:var(--muted);border-left:3px solid var(--line);padding-left:10px;margin:0 0 18px}
@media (prefers-color-scheme:dark){.b-partial{color:#fbbf24}}
</style>
</head>
<body>
<main>
<h1>Question answering over FDA drug labels</h1>
<p class="lede">A retrieval-augmented system answers questions using only excerpts from ~1,200 official FDA drug labels and cites them.
Browse every evaluation question with the answer from each retrieval setup, the excerpts behind it, and an LLM judge's verdicts.
<a href="https://github.com/krishnagiri1/drug-rag">Code and write-up on GitHub</a>.</p>
<p class="disclaimer">For information and research only. Not medical advice.</p>
<nav role="tablist">
<button role="tab" aria-selected="true" data-tab="browse">Browse answers</button>
<button role="tab" aria-selected="false" data-tab="results">Results</button>
</nav>
<section id="browse">
<div class="controls">
<label>Retrieval setup<select id="config"></select></label>
<label>Show<select id="filter">
<option value="all">All questions</option>
<option value="correct">Judged correct</option>
<option value="partial">Judged partial</option>
<option value="incorrect">Judged incorrect</option>
<option value="refused">Refused</option>
<option value="unsupported">Has an unsupported claim</option>
<option value="diff">Setups disagree</option>
</select></label>
<label>Question<select id="pick"></select></label>
<div class="nav2"><button id="prev" aria-label="Previous">‹ Prev</button><button id="next" aria-label="Next">Next ›</button></div>
<span class="count" id="count"></span>
</div>
<div id="view"></div>
</section>
<section id="results" hidden><div id="report"></div></section>
<footer>Data: openFDA drug labels (U.S. FDA public data). Built by Venkata Krishna Giri Koneru.</footer>
</main>
<script src="data.js"></script>
<script>
const D = window.DRUG_RAG_DATA;
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
$("config").innerHTML = D.configs.map((c, i) => `<option value="${c.key}">${esc(c.name)}</option>`).join("");
let list = [];
function status(a) { return !a ? null : a.refused ? "refused" : a.verdict; }
function applyFilter() {
  const cfg = $("config").value, f = $("filter").value;
  list = D.items.filter(it => {
    const a = it.answers[cfg]; if (!a) return false;
    if (f === "all") return true;
    if (f === "unsupported") return a.unsupported.length > 0;
    if (f === "diff") { const s = D.configs.map(c => status(it.answers[c.key])).filter(Boolean); return new Set(s).size > 1; }
    return status(a) === f;
  });
  $("pick").innerHTML = list.map((it, i) => `<option value="${i}">${esc(it.question.slice(0, 90))}${it.question.length > 90 ? "…" : ""}</option>`).join("");
  $("count").textContent = `${list.length} question${list.length === 1 ? "" : "s"}`;
  show(0);
}
function badge(a) {
  const s = status(a); if (!s) return "";
  const label = {correct: "✓ correct", partial: "◐ partial", incorrect: "✗ incorrect", refused: "refused"}[s] || s;
  return `<span class="badge b-${s}">${label}</span>`;
}
function show(i) {
  if (!list.length) { $("view").innerHTML = "<p>No questions match this filter.</p>"; return; }
  i = Math.max(0, Math.min(list.length - 1, i)); $("pick").value = i;
  const it = list[i], cfg = $("config").value, a = it.answers[cfg];
  let h = `<div class="card"><h3>${esc(it.question)}</h3><div class="meta">Expected source: ${esc(it.drug)} · ${esc(it.section)} · ${it.kind === "template" ? "template question" : "LLM-written, human-reviewed question"}</div>`;
  if (it.reference) h += `<p class="meta"><b>${it.kind === "template" ? "Start of the expected label section" : "Reference answer"}:</b> ${esc(it.reference.slice(0, 600))}${it.reference.length > 600 ? "…" : ""}</p>`;
  h += `</div>`;
  h += `<div class="card"><h3>Answer ${badge(a)}${a.faithfulness != null && !a.refused ? `<span class="badge">${Math.round(a.faithfulness * 100)}% of claims supported</span>` : ""}</h3>`;
  h += `<p class="answer">${esc(a.answer)}</p>`;
  if (a.reason) h += `<p class="meta"><b>Judge:</b> ${esc(a.reason)}</p>`;
  for (const u of a.unsupported) h += `<p class="unsup">Unsupported claim: ${esc(u)}</p>`;
  h += `<p class="meta">Right passage retrieved: ${a.gold_retrieved ? "yes" : "no"}</p></div>`;
  h += `<div class="card"><h3>Retrieved label excerpts</h3>`;
  for (const s of a.sources) h += `<div class="src"><div class="head">[${s.n}] ${esc(s.drug)} — ${esc(s.section)} ${s.cited ? '<span class="cited">· cited</span>' : ""}</div><p>${esc(s.text)}</p></div>`;
  h += `</div>`;
  $("view").innerHTML = h;
}
$("config").onchange = applyFilter; $("filter").onchange = applyFilter;
$("pick").onchange = e => show(+e.target.value);
$("prev").onclick = () => show(+$("pick").value - 1);
$("next").onclick = () => show(+$("pick").value + 1);
document.querySelectorAll("nav button").forEach(b => b.onclick = () => {
  document.querySelectorAll("nav button").forEach(x => x.setAttribute("aria-selected", x === b));
  $("browse").hidden = b.dataset.tab !== "browse"; $("results").hidden = b.dataset.tab !== "results";
});
$("report").innerHTML = D.report || "<p>No report yet.</p>";
applyFilter();
</script>
</body>
</html>
"""


def export_site(out_dir: str | Path = "docs", results_dir: Path = RESULTS_DIR, testset_path: Path = TESTSET_PATH,
                index_root: Path = INDEX_DIR) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = build_data(results_dir, testset_path, index_root)
    (out / "data.js").write_text("window.DRUG_RAG_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    (out / "index.html").write_text(PAGE.replace("__CSS__", BASE_CSS), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return out / "index.html"
