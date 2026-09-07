"""Publish source-rendered guides at established GitHub Pages URLs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from bs4 import BeautifulSoup

ORIGIN = 'https://www.funasr.com'
ALIASES = {
    'quickstart': 'tutorial.html',
    'training': 'training.html',
    'model-registration': 'model-registration.html',
    'model-selection': 'model-selection.html',
    'moss-transcribe-diarize': 'moss-transcribe-diarize.html',
    'vllm': 'vllm.html',
    'deployment-matrix': 'deployment-matrix.html',
}


def insert_legacy_anchors(soup: BeautifulSoup, mapping: dict[str, str]) -> None:
    """Place legacy fragments next to their current section without changing it."""
    article = soup.select_one('.docs-article')
    if article is None:
        # Non-catalogue HTML has no source-article heading contract.
        return
    for old, target in mapping.items():
        headings = article.find_all(id=target)
        if len(headings) != 1 or headings[0].name not in ('h2', 'h3'):
            raise ValueError(f'Legacy fragment {old!r} needs one h2/h3 target {target!r}')
        existing = soup.find_all(id=old)
        if len(existing) > 1:
            raise ValueError(f'Duplicate legacy fragment {old!r}')
        if existing:
            continue
        anchor = soup.new_tag('span', id=old)
        anchor['data-legacy-doc-anchor'] = target
        headings[0].insert_before(anchor)


def export_documentation(built: Path, output: Path, aliases: dict | None = None) -> None:
    """Keep API/hub pages intact and publish immutable assets with the guides."""
    legacy = json.loads((Path(__file__).parent / 'data/legacy_doc_anchors.json').read_text(encoding='utf-8'))
    if (built / 'assets').is_dir():
        shutil.copytree(built / 'assets', output / 'assets', dirs_exist_ok=True)
    for slug, filename in (ALIASES if aliases is None else aliases).items():
        for source_prefix, target_prefix in (('en/', ''), ('', 'zh/')):
            source = built / source_prefix / 'docs' / f'{slug}.html'
            soup = BeautifulSoup(source.read_text(encoding='utf-8'), 'html.parser')
            insert_legacy_anchors(soup, legacy['pages'].get(target_prefix + filename, {}))
            for node in soup.find_all(True):
                for attribute in ('href', 'src', 'action', 'data-copy-icon'):
                    value = node.get(attribute, '')
                    if isinstance(value, str) and value.startswith('/') and not value.startswith('//'):
                        if value.startswith('/assets/'):
                            node[attribute] = ('../' if target_prefix else '') + value.lstrip('/')
                        else:
                            node[attribute] = ORIGIN + value
            target = output / target_prefix / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(soup), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_documentation(args.site, args.output)
