from pathlib import Path
import json
import sys
from urllib.parse import unquote

from bs4 import BeautifulSoup
import pytest

SITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE_ROOT))
from build import build


@pytest.fixture(scope='module')
def output(tmp_path_factory):
    path = tmp_path_factory.mktemp('documentation')
    build(path)
    return path


def test_bilingual_docs_have_navigation_and_real_source_content(output):
    for prefix in ('', 'en/'):
        index = BeautifulSoup((output / prefix / 'docs/index.html').read_text(), 'html.parser')
        assert index.select_one('[data-doc-search]')
        assert index.select_one('a[href$="/docs/moss-transcribe-diarize.html"]')
        moss = BeautifulSoup((output / prefix / 'docs/moss-transcribe-diarize.html').read_text(), 'html.parser')
        assert moss.select_one('.docs-sidebar')
        assert moss.select_one('.docs-toc a[href^="#"]')
        assert 'OpenMOSS' in moss.select_one('.docs-article').get_text()
        assert moss.select_one('.docs-article pre code')
        assert moss.select_one('[data-source-link]')['href'].endswith('.md')


def test_official_native_record_has_local_language_links_and_search(output):
    for language, prefix, suffix in (('zh', '', '_zh'), ('en', 'en/', '')):
        route = '/' + prefix + 'docs/official-native-vllm.html'
        page = BeautifulSoup((output / route.lstrip('/')).read_text(), 'html.parser')
        assert page.select_one('[data-source-link]')['href'].endswith(
            f'/docs/vllm_official_native_validation{suffix}.md')
        links = {link['href'] for link in page.select('.docs-article a[href]')}
        assert '/' + prefix + 'docs/native-vllm.html' in links
        assert '/' + prefix + 'docs/deployment-matrix.html' in links
        assert any('/docs/benchmark/vllm_official_native_20260907.json' in link for link in links)
        rows = json.loads((output / f'search-{language}.json').read_text())
        assert any(row['url'] == route and '2026-09-07' in row['title'] for row in rows)
        deploy = BeautifulSoup((output / prefix / 'deploy/vllm.html').read_text(), 'html.parser')
        commands = deploy.select_one('[data-section="commands"]').get_text()
        assert 'FunAudioLLM/Fun-ASR-Nano-2512-vllm' in commands
        assert 'allendou/' not in commands


def test_deployment_limitations_use_short_heading_and_preserve_full_body(output):
    entries = json.loads((SITE_ROOT / 'data/deployments.json').read_text())['deployments']
    for language in ('zh', 'en'):
        for entry in entries:
            route = entry['routes'][language].lstrip('/')
            page = BeautifulSoup((output / route).read_text(), 'html.parser')
            callout = page.select_one('[data-section="limitations"] .limitation-callout')
            heading = callout.select_one('h2')
            expected = entry['translations'][language]['primary_limitation']
            assert heading.get_text() == ('已知限制' if language == 'zh' else 'Known limitations')
            assert heading.get_text() != expected
            summary = callout.select_one('p.limitation-summary')
            assert summary is not None, (language, entry['id'])
            assert summary.get_text() == expected, (language, entry['id'])
            assert not callout.select('.eyebrow')


def test_search_index_is_bilingual_and_links_to_emitted_pages(output):
    for language in ('zh', 'en'):
        rows = json.loads((output / f'search-{language}.json').read_text())
        assert len(rows) >= 12
        assert any('MOSS' in row['title'] for row in rows)
        for row in rows:
            path = row['url'].split('#')[0].lstrip('/')
            if path.endswith('/'):
                path += 'index.html'
            assert (output / path).is_file(), row['url']


def test_legacy_articles_and_new_pages_share_versioned_design(output):
    for relative in ('index.html', 'models.html', 'blog/index.html', 'donors.html', 'deploy/vllm.html'):
        soup = BeautifulSoup((output / relative).read_text(), 'html.parser')
        assert soup.select_one('link[href*="experience."][href$=".css"]'), relative
        assert soup.select_one('[data-primary-nav]'), relative
        links = soup.select('[data-primary-nav] a')
        assert links[-1]['href'].endswith('donors.html')
    home = (output / 'index.html').read_text()
    assert 'data-doc-search' in home
    assert 'Apache-2.0</dd>' not in home


def test_documentation_links_are_mapped_to_local_pages_or_repository(output):
    soup = BeautifulSoup((output / 'docs/model-selection.html').read_text(), 'html.parser')
    for link in soup.select('.docs-article a[href]'):
        href = link['href']
        assert href.startswith(('/', '#', 'https://', 'http://', 'mailto:')), href


def test_source_fragment_links_keep_unicode_and_punctuation_boundaries(output):
    for prefix in ('', 'en/'):
        for path in (output / prefix / 'docs').glob('*.html'):
            soup = BeautifulSoup(path.read_text(), 'html.parser')
            ids = {node['id'] for node in soup.select('[id]')}
            for link in soup.select('.docs-article a[href^="#"]'):
                assert unquote(link['href'][1:]) in ids, (path, link['href'])


def test_section_based_quickstarts_are_searchable(output):
    for language, route in (('zh', '/quickstart.html'), ('en', '/en/quickstart.html')):
        rows = json.loads((output / f'search-{language}.json').read_text())
        assert any(row['url'] == route for row in rows)


def test_github_pages_hubs_have_task_navigation_and_search_handoff():
    root = SITE_ROOT.parents[1]
    for relative, destination in (('index.html', '/en/docs/'), ('zh/index.html', '/docs/')):
        soup = BeautifulSoup((root / 'gh-pages-output' / relative).read_text(), 'html.parser')
        assert len(soup.select('.chapters section')) == 4
        assert soup.select_one('form[role="search"]')['action'] == 'https://www.funasr.com' + destination
        assert soup.select_one('a[href$="moss-transcribe-diarize.html"]')
        assert soup.select_one('a[href$="api.html"]')
        assert '170x' not in soup.get_text()
