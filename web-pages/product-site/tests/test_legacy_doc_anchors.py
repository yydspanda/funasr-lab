"""Compatibility contracts for the published pre-catalogue guide URLs."""

import json
from pathlib import Path
import sys

from bs4 import BeautifulSoup
import pytest

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE))

# IDs independently inventoried from gh-pages at 871ac766876c2f511550478d88c86aa3daab3172.
OLD_IDS = {
    'tutorial.html': 'quick install choose scenarios offline streaming spk emotion vad punc deploy-agent subtitle export faq',
    'training.html': 'overview data paraformer sensevoice nano params multi-gpu deepspeed monitor inference-after tips',
    'model-registration.html': 'architecture setup understand add-model add-component standalone testing pitfalls contribute',
    'model-selection.html': 'default decision aliases runtime benchmark',
    'moss-transcribe-diarize.html': 'contract server runtimes evidence boundaries',
    'vllm.html': 'overview install offline streaming-sdk websocket dynamic-vad performance api-reference faq',
    'deployment-matrix.html': 'matrix choices runtime-diagnostics checklist help',
}
PAGES = [prefix + name for name in OLD_IDS for prefix in ('', 'zh/')]


@pytest.fixture(scope='module')
def exported(tmp_path_factory):
    from documentation import load_catalogue, render_source
    from export_docs import ALIASES, export_documentation

    root = tmp_path_factory.mktemp('legacy-docs')
    built, output = root / 'built', root / 'pages'
    catalogue = load_catalogue()
    entries = {entry['slug']: entry for entry in catalogue['pages']}
    originals = {}
    for slug, filename in ALIASES.items():
        for language, source_prefix, target_prefix in (('en', 'en/', ''), ('zh', '', 'zh/')):
            content = render_source(entries[slug], language, catalogue)['content_html']
            html = ('<html><head><link rel="stylesheet" href="/assets/site.123.css">'
                    '<script src="/assets/experience.456.js" defer></script></head><body>'
                    '<a href="/docs/python-api.html">API</a>'
                    f'<article class="docs-article">{content}</article></body></html>')
            path = built / source_prefix / 'docs' / f'{slug}.html'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding='utf-8')
            originals[target_prefix + filename] = (path, html)
    (built / 'assets').mkdir()
    (built / 'assets/site.123.css').write_text('body { color: black; }')
    (built / 'assets/experience.456.js').write_text('"use strict";')
    for name in ('api.html', 'zh/api.html', 'reference/AutoModel.html', 'index.html'):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'existing reference\n')
    export_documentation(built, output)
    return built, output, originals


@pytest.mark.parametrize('page', PAGES)
def test_every_published_fragment_survives(exported, page):
    _, output, _ = exported
    soup = BeautifulSoup((output / page).read_text(), 'html.parser')
    ids = [node['id'] for node in soup.select('[id]')]
    expected = set(OLD_IDS[Path(page).name].split())
    assert expected <= set(ids), (page, sorted(expected - set(ids)))
    assert len(ids) == len(set(ids)), page


@pytest.mark.parametrize('page', PAGES)
def test_aliases_precede_their_semantic_headings(exported, page):
    _, output, _ = exported
    manifest = json.loads((SITE / 'data/legacy_doc_anchors.json').read_text())
    mapping = manifest['pages'][page]
    assert set(mapping) == set(OLD_IDS[Path(page).name].split())
    soup = BeautifulSoup((output / page).read_text(), 'html.parser')
    article = soup.select_one('.docs-article')
    for old, target in mapping.items():
        anchor = article.find(id=old)
        heading = article.find(id=target)
        assert anchor is not None and heading is not None, (page, old, target)
        assert heading.name in ('h2', 'h3'), (page, target)
        assert anchor.name == 'span' and not anchor.get_text(), (page, old)
        sibling = anchor.find_next_sibling()
        while sibling is not None and sibling.name == 'span' and sibling.get('data-legacy-doc-anchor') is not None:
            sibling = sibling.find_next_sibling()
        assert sibling is heading, (page, old, target)


def test_export_preserves_source_content_api_and_local_assets(exported):
    _, output, originals = exported
    for page, (path, original) in originals.items():
        assert path.read_text() == original
        before = BeautifulSoup(original, 'html.parser')
        after = BeautifulSoup((output / page).read_text(), 'html.parser')
        assert after.select_one('.docs-article').get_text() == before.select_one('.docs-article').get_text()
        assert [(n.name, n['id'], n.get_text()) for n in after.select('h2[id],h3[id]')] == [
            (n.name, n['id'], n.get_text()) for n in before.select('h2[id],h3[id]')]
        assert [n.get_text() for n in after.select('pre')] == [n.get_text() for n in before.select('pre')]
        assert after.select_one('a[href="https://www.funasr.com/docs/python-api.html"]')
        for node, attribute in ((after.select_one('link[rel=stylesheet]'), 'href'), (after.script, 'src')):
            assert not node[attribute].startswith(('/', 'http'))
            assert ((output / page).parent / node[attribute]).is_file()
        for node in after.select('[data-legacy-doc-anchor]'):
            node.decompose()
        # Apart from the existing root-link rewrite, the article DOM is unchanged.
        for node in before.select('[href], [src]'):
            for attribute in ('href', 'src'):
                value = node.get(attribute, '')
                if value.startswith('/') and not value.startswith('//'):
                    node[attribute] = 'https://www.funasr.com' + value
        assert str(after.select_one('.docs-article')) == str(before.select_one('.docs-article'))
    for name in ('api.html', 'zh/api.html', 'reference/AutoModel.html', 'index.html'):
        assert (output / name).read_bytes() == b'existing reference\n'


def test_training_alias_and_moss_mappings_are_topic_specific():
    pages = json.loads((SITE / 'data/legacy_doc_anchors.json').read_text())['pages']
    for prefix, data, params, aliases, server in (
        ('', 'prepare-the-dataset', 'validate-small-then-launch', 'openai-compatible-api-aliases', 'funasr-openai-compatible-service'),
        ('zh/', '准备数据', '先做小规模验证', 'openai-兼容-api-别名', 'funasr-openai-兼容服务'),
    ):
        assert pages[prefix + 'training.html']['data'] == data
        assert pages[prefix + 'training.html']['params'] == params
        assert pages[prefix + 'model-selection.html']['aliases'] == aliases
        assert pages[prefix + 'moss-transcribe-diarize.html']['server'] == server
        assert pages[prefix + 'moss-transcribe-diarize.html']['evidence'] == server


def test_insertion_is_idempotent_and_keeps_existing_anchors():
    from export_docs import insert_legacy_anchors

    soup = BeautifulSoup('<article class="docs-article"><a id="existing"></a><h2 id="topic">Topic</h2></article>', 'html.parser')
    mapping = {'old': 'topic', 'existing': 'topic'}
    insert_legacy_anchors(soup, mapping)
    first = str(soup)
    insert_legacy_anchors(soup, mapping)
    assert str(soup) == first
    assert len(soup.select('#existing')) == 1
    assert soup.find(id='existing').name == 'a'


@pytest.mark.parametrize('heading', ('<h2 id="different">Topic</h2>', '<p id="topic">Not a heading</p>'))
def test_missing_or_nonheading_target_fails_explicitly(heading):
    from export_docs import insert_legacy_anchors

    soup = BeautifulSoup(f'<article class="docs-article">{heading}</article>', 'html.parser')
    with pytest.raises(ValueError, match='topic'):
        insert_legacy_anchors(soup, {'old': 'topic'})
