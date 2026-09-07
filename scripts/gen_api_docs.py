#!/usr/bin/env python3
"""Generate API page with source code preview and GitHub links.
Run this script from the repo root on main branch.
Output: gh-pages-output/api.html for deployment to gh-pages.

This script is designed to be run by GitHub Actions on every push to main,
automatically regenerating the API docs.
"""
import argparse
import ast
import glob
import html as htmlmod
import os
import re
import shutil
from pathlib import Path

REPO_URL = "https://github.com/modelscope/FunASR"
BRANCH = "main"
REPO_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "gh-pages-output")
OUTPUT_DIR = parser.parse_args().output_dir.resolve()
STYLESHEET = REPO_ROOT / "gh-pages-output" / "api-reference.css"

os.chdir(REPO_ROOT)

tree_structure = {
    "funasr.auto": {"auto_model": "funasr/auto/auto_model.py"},
    "funasr.register": {"register": "funasr/register.py"},
    "funasr.models": {},
    "funasr.frontends": {},
    "funasr.tokenizer": {},
    "funasr.utils": {},
    "funasr.bin": {},
    "funasr.download": {},
    "funasr.train_utils": {},
    "funasr.datasets": {},
    "funasr.losses": {},
    "funasr.schedulers": {},
}

# Auto-discover
for f in sorted(glob.glob("funasr/models/*/model.py")):
    if "whisper_lib" in f:
        continue
    tree_structure["funasr.models"][f.split("/")[-2]] = f

for top in ["frontends", "tokenizer", "utils", "bin", "download", "train_utils", "losses", "schedulers"]:
    for f in sorted(glob.glob(f"funasr/{top}/*.py")):
        if "__init__" in f or "abs_" in f:
            continue
        tree_structure[f"funasr.{top}"][os.path.basename(f).replace(".py", "")] = f

for f in sorted(glob.glob("funasr/datasets/*.py") + glob.glob("funasr/datasets/audio_datasets/*.py")):
    if "__init__" in f or "__pycache__" in f:
        continue
    tree_structure["funasr.datasets"][os.path.basename(f).replace(".py", "")] = f

tree_structure = {k: v for k, v in tree_structure.items() if v}

SKIP_CLASSES = {
    "SinusoidalPositionEncoder", "StreamSinusoidalPositionEncoder",
    "PositionwiseFeedForward", "MultiHeadedAttentionSANM", "LayerNorm",
    "EncoderLayerSANM", "VadStateMachine", "FrameState", "AudioChangeState",
    "VadDetectMode", "VADXOptions", "E2EVadSpeechBufWithDoa",
    "E2EVadFrameProb", "WindowDetector", "Stats",
}

def esc(s):
    return htmlmod.escape(s) if s else ""

def format_doc(doc):
    if not doc:
        return '<p class="muted">No documentation yet.</p>'
    lines = doc.strip().split("\n")
    out = ""
    in_list = False
    for line in lines:
        s = line.strip()
        if s in ("Args:", "Returns:", "Raises:", "Examples:", "Note:", "Notes:", "Features:", "Models:", "Requirements:", "Output:", "Output format:"):
            if in_list: out += "</ul>\n"; in_list = False
            out += f'<h4>{s}</h4>\n'
            continue
        if s.startswith("- "):
            if not in_list: out += '<ul>\n'; in_list = True
            out += f'<li>{esc(s[2:])}</li>\n'
            continue
        m = re.match(r'^(\*{0,2}\w[\w.*]*(?:\s*\([^)]*\))?)\s*[:—\-]\s*(.+)', s)
        if m and not s.startswith("http"):
            if not in_list: out += '<ul>\n'; in_list = True
            out += f'<li><code>{esc(m.group(1))}</code> — {esc(m.group(2))}</li>\n'
            continue
        if (line.startswith("        ") or line.startswith("            ")) and in_list and s:
            out += f'<li class="sub">{esc(s)}</li>\n'
            continue
        if s:
            if in_list: out += "</ul>\n"; in_list = False
            out += f'<p>{esc(s)}</p>\n'
    if in_list: out += "</ul>\n"
    return out

def get_source_lines(filepath, node):
    """Get source code lines for an AST node."""
    with open(filepath) as f:
        all_lines = f.readlines()
    start = node.lineno - 1
    end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start + 20
    # Limit preview to 30 lines
    preview_end = min(start + 30, end)
    source = "".join(all_lines[start:preview_end])
    truncated = preview_end < end
    return source, start + 1, truncated

def github_url(filepath, lineno):
    return f"{REPO_URL}/blob/{BRANCH}/{filepath}#L{lineno}"

# Extract all data
all_data = {}
for top_level, sub_modules in tree_structure.items():
    for sub_name, filepath in sub_modules.items():
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            source = f.read()
        try:
            tree = ast.parse(source, filename=filepath)
        except:
            continue
        key = f"{top_level}.{sub_name}"
        entries = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                if node.name in SKIP_CLASSES:
                    continue
                class_doc = ast.get_docstring(node) or ""
                src, lineno, trunc = get_source_lines(filepath, node)
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name.startswith("_"):
                            continue
                        mdoc = ast.get_docstring(item) or ""
                        args = [a.arg for a in item.args.args if a.arg != "self"]
                        if item.args.vararg: args.append("*" + item.args.vararg.arg)
                        if item.args.kwarg: args.append("**" + item.args.kwarg.arg)
                        msrc, mline, mtrunc = get_source_lines(filepath, item)
                        methods.append({"name": item.name, "args": args, "doc": mdoc,
                                       "source": msrc, "lineno": mline, "truncated": mtrunc})
                entries.append({"type": "class", "name": node.name, "doc": class_doc,
                               "methods": methods, "source": src, "lineno": lineno,
                               "truncated": trunc, "filepath": filepath})
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                fdoc = ast.get_docstring(node) or ""
                args = [a.arg for a in node.args.args]
                if node.args.vararg: args.append("*" + node.args.vararg.arg)
                if node.args.kwarg: args.append("**" + node.args.kwarg.arg)
                fsrc, fline, ftrunc = get_source_lines(filepath, node)
                entries.append({"type": "function", "name": node.name, "args": args,
                               "doc": fdoc, "source": fsrc, "lineno": fline,
                               "truncated": ftrunc, "filepath": filepath})
        if entries:
            all_data[key] = entries

total_entries = sum(len(e) for e in all_data.values())
print(f"Extracted: {len(all_data)} modules, {total_entries} entries")

# Generate HTML
sidebar = ""
content = ""
overview_links = ""
eid = 0

for group_index, top_level in enumerate(tree_structure):
    sub_modules = tree_structure[top_level]
    group_id = f"group-{group_index}"
    sidebar += f'<div class="l1"><button type="button" class="l1-title" aria-expanded="true" aria-controls="{group_id}"><span>{esc(top_level)}</span><span class="cnt">{sum(1 for s in sub_modules if f"{top_level}.{s}" in all_data)}</span><span class="arr" aria-hidden="true">▸</span></button><div class="l1-children" id="{group_id}">\n'
    for module_index, sub_name in enumerate(sub_modules):
        key = f"{top_level}.{sub_name}"
        if key not in all_data:
            continue
        module_id = f"{group_id}-module-{module_index}"
        sidebar += f'<div class="l2"><button type="button" class="l2-title" aria-expanded="true" aria-controls="{module_id}"><span>{esc(sub_name)}</span><span class="arr" aria-hidden="true">▸</span></button><div class="l2-children" id="{module_id}">\n'
        for entry in all_data[key]:
            eid += 1
            eid_str = f"e{eid}"
            filepath = entry.get("filepath", "")
            if entry["type"] == "class":
                if entry["name"] in {"AutoModel", "RegisterTables"}:
                    overview_links += f'<a href="#{eid_str}" class="entry-link">{esc(entry["name"])} <span aria-hidden="true">→</span></a> '
                sidebar += f'<a class="l3-item" href="#{eid_str}" data-target="{eid_str}"><span class="mb badge-c" aria-hidden="true">C</span><span>{esc(entry["name"])}</span></a>\n'
                gh_link = github_url(filepath, entry["lineno"])
                content += f'<div class="api-detail" id="{eid_str}"><span class="dtag badge-c">class</span><h2>{esc(entry["name"])}</h2><div class="dmod">{key} · <a href="{gh_link}" target="_blank">View on GitHub ↗</a></div><div class="ddoc">{format_doc(entry["doc"])}</div>'
                content += f'<details class="src-block"><summary>📄 Source code</summary><pre><code>{esc(entry["source"])}</code></pre>'
                if entry["truncated"]:
                    content += f'<a href="{gh_link}" target="_blank" class="src-more">View full source on GitHub →</a>'
                content += '</details>'
                if entry["methods"]:
                    content += '<h3 class="mt">Methods</h3>'
                    for m in entry["methods"]:
                        sig = f'.{m["name"]}({", ".join(m["args"])})'
                        mgh = github_url(filepath, m["lineno"])
                        content += f'<div class="mblk"><code>{esc(sig)}</code> <a href="{mgh}" target="_blank" class="gh-link">L{m["lineno"]}</a><div class="mdoc">{format_doc(m["doc"])}</div>'
                        content += f'<details class="src-block"><summary>📄 Source</summary><pre><code>{esc(m["source"])}</code></pre>'
                        if m["truncated"]:
                            content += f'<a href="{mgh}" target="_blank" class="src-more">View full source →</a>'
                        content += '</details></div>'
                content += '</div>\n'
                for m in entry["methods"]:
                    eid += 1
                    mid = f"e{eid}"
                    sidebar += f'<a class="l3-item l3m" href="#{mid}" data-target="{mid}"><span class="mb badge-m" aria-hidden="true">M</span><span>.{esc(m["name"])}()</span></a>\n'
                    sig = f'{entry["name"]}.{m["name"]}({", ".join(m["args"])})'
                    mgh = github_url(filepath, m["lineno"])
                    content += f'<div class="api-detail" id="{mid}"><span class="dtag badge-m">method</span><h2><code>{esc(sig)}</code></h2><div class="dmod">{key}.{entry["name"]} · <a href="{mgh}" target="_blank">View on GitHub ↗</a></div><div class="ddoc">{format_doc(m["doc"])}</div>'
                    content += f'<details class="src-block"><summary>📄 Source code</summary><pre><code>{esc(m["source"])}</code></pre>'
                    if m["truncated"]:
                        content += f'<a href="{mgh}" target="_blank" class="src-more">View full source on GitHub →</a>'
                    content += '</details></div>\n'
            else:
                args_str = ", ".join(entry.get("args", []))
                gh_link = github_url(filepath, entry["lineno"])
                sidebar += f'<a class="l3-item" href="#{eid_str}" data-target="{eid_str}"><span class="mb badge-f" aria-hidden="true">F</span><span>{esc(entry["name"])}</span></a>\n'
                content += f'<div class="api-detail" id="{eid_str}"><span class="dtag badge-f">function</span><h2><code>{esc(entry["name"])}({esc(args_str)})</code></h2><div class="dmod">{key} · <a href="{gh_link}" target="_blank">View on GitHub ↗</a></div><div class="ddoc">{format_doc(entry["doc"])}</div>'
                content += f'<details class="src-block"><summary>📄 Source code</summary><pre><code>{esc(entry["source"])}</code></pre>'
                if entry["truncated"]:
                    content += f'<a href="{gh_link}" target="_blank" class="src-more">View full source on GitHub →</a>'
                content += '</details></div>\n'
        sidebar += '</div></div>\n'
    sidebar += '</div></div>\n'

# Preserve extracted content and entry numbering while adding focus targets.
content = content.replace('class="api-detail"', 'class="api-detail" tabindex="-1"')
content = content.replace('target="_blank"', 'target="_blank" rel="noopener noreferrer"')
content = content.replace('<summary>📄 Source', '<summary>Source')

html_page = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>FunASR API</title>
<link rel="stylesheet" href="api-reference.css">
</head><body class="api-page">
<a class="skip-link" href="#api-content">Skip to content</a>
<nav class="nav" aria-label="Main navigation"><div class="container">
<a href="index.html" class="nav-logo">FunASR</a>
<div class="nav-links">
                <a href="index.html">Home</a>
                <a href="tutorial.html">Tutorial</a>
                <a href="training.html">Training</a>
                <a href="model-registration.html">Develop</a>
                <a href="api.html" class="active" aria-current="page">API</a>
            </div>
<a href="https://github.com/modelscope/FunASR" class="nav-github">GitHub</a>
</div></nav>
<header class="api-header"><h1>API Reference</h1><p>{len(all_data)} modules <span aria-hidden="true">/</span> {total_entries} classes and functions</p></header>
<div class="api-layout">
<aside class="api-sidebar" aria-label="API navigation">
<label class="sr-only" for="api-search">Search API entries</label>
<input id="api-search" type="search" class="sb-search" placeholder="Search API" autocomplete="off" spellcheck="false">
<p id="search-empty" class="search-empty" role="status" hidden>No matching APIs</p>
<nav aria-label="Modules and entries">
{sidebar}
</nav></aside>
<main class="api-content" id="api-content" tabindex="-1">
<div class="api-welcome" id="api-welcome"><h2>FunASR Python API</h2><p class="muted">Models, inference, training, and utilities.</p>{overview_links}</div>
{content}
</main></div>
<script>
const entries = Array.from(document.querySelectorAll('.api-detail'));
const entryLinks = Array.from(document.querySelectorAll('.l3-item'));
const controls = Array.from(document.querySelectorAll('.l1-title, .l2-title'));
const welcome = document.getElementById('api-welcome');
const search = document.getElementById('api-search');
let expandedBeforeSearch = null;

function setExpanded(button, expanded) {{
    button.setAttribute('aria-expanded', String(expanded));
    document.getElementById(button.getAttribute('aria-controls')).hidden = !expanded;
}}
function showEntry(id, focus = false) {{
    // Resolve only known entries, never interpolate an untrusted hash in a selector.
    const target = entries.find(entry => entry.id === id) || null;
    entries.forEach(entry => {{ entry.hidden = entry !== target; }});
    welcome.hidden = Boolean(target);
    entryLinks.forEach(link => {{
        if (target && link.dataset.target === target.id) {{
            link.setAttribute('aria-current', 'true');
            if (link.hidden) {{ search.value = ''; filterSidebar(''); }}
            for (const level of ['.l1', '.l2']) {{
                const group = link.closest(level);
                if (group) setExpanded(group.querySelector('button'), true);
            }}
        }} else {{
            link.removeAttribute('aria-current');
        }}
    }});
    document.title = target ? target.querySelector('h2').textContent + ' | FunASR API' : 'FunASR API';
    if (focus && target) {{
        target.focus({{preventScroll: true}});
        target.scrollIntoView({{block: 'start'}});
    }}
}}
function filterSidebar(query) {{
    const q = query.trim().toLowerCase();
    if (q && !expandedBeforeSearch) {{
        expandedBeforeSearch = new Map(controls.map(button => [button, button.getAttribute('aria-expanded') === 'true']));
    }}
    entryLinks.forEach(link => {{ link.hidden = Boolean(q) && !link.textContent.toLowerCase().includes(q); }});
    document.querySelectorAll('.l2, .l1').forEach(group => {{
        group.hidden = Boolean(q) && !Array.from(group.querySelectorAll('.l3-item')).some(link => !link.hidden);
    }});
    if (q) controls.forEach(button => setExpanded(button, true));
    else if (expandedBeforeSearch) {{
        expandedBeforeSearch.forEach((expanded, button) => setExpanded(button, expanded));
        expandedBeforeSearch = null;
    }}
    document.getElementById('search-empty').hidden = !q || entryLinks.some(link => !link.hidden);
}}
controls.forEach(button => {{
    setExpanded(button, false);
    button.addEventListener('click', () => setExpanded(button, button.getAttribute('aria-expanded') !== 'true'));
}});
entryLinks.forEach(link => link.addEventListener('click', event => {{
    if (!event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey && event.button === 0) {{
        showEntry(link.dataset.target, true);
    }}
}}));
search.addEventListener('input', () => filterSidebar(search.value));
document.querySelector('.skip-link').addEventListener('click', event => {{
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button !== 0) return;
    event.preventDefault();
    document.getElementById('api-content').focus();
}});
window.addEventListener('hashchange', () => showEntry(window.location.hash.slice(1)));
showEntry(window.location.hash.slice(1));
</script></body></html>'''

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if STYLESHEET.resolve() != (OUTPUT_DIR / STYLESHEET.name).resolve():
    shutil.copy2(STYLESHEET, OUTPUT_DIR / STYLESHEET.name)
with (OUTPUT_DIR / "api.html").open("w", encoding="utf-8") as f:
    f.write(html_page)

training_src = REPO_ROOT / "training.html"
if training_src.exists():
    shutil.copy2(training_src, OUTPUT_DIR / "training.html")

print(f"Generated {OUTPUT_DIR / 'api.html'}")
