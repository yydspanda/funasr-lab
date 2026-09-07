"""Source-backed service docs checks without loading models or starting servers."""

import ast
import json
from pathlib import Path
import re
import shlex
import subprocess
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/openai_api"
DOCS = [EXAMPLE / name for name in ("CLIENTS.md", "README.md", "README_zh.md")]
PACKAGED = ROOT / "funasr/bin/_server_app.py"


def blocks(text, language):
    return re.findall(r"^```" + language + r"\n(.*?)^```", text, re.M | re.S)


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def function(path, name):
    return next(node for node in ast.walk(tree(path))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name)


def form_defaults(node):
    defaults = {}
    for arg, value in zip(node.args.args[-len(node.args.defaults):], node.args.defaults):
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "Form":
            defaults[arg.arg] = ast.literal_eval(next(
                keyword.value for keyword in value.keywords if keyword.arg == "default"))
    return defaults


def headings(text):
    text = re.sub(r"^```[^\n]*\n.*?^```[^\n]*$", "", text, flags=re.M | re.S)
    result = set()
    counts = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        result.add(f"{slug}-{count}" if count else slug)
    return result


class ServiceDocsContract(unittest.TestCase):
    def test_relative_links_and_owned_page_anchors(self):
        owned = {path.resolve() for path in DOCS}
        for path in DOCS:
            text = path.read_text(encoding="utf-8")
            self.assertIn("## API Contract", text)
            self.assertIn("](#api-contract)", text)
            for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                parts = urlsplit(link)
                if parts.scheme or parts.netloc:
                    continue
                target = (path.parent / unquote(parts.path)).resolve() if parts.path else path.resolve()
                self.assertIn(ROOT.resolve(), target.parents)
                self.assertNotIn(".mcp-tasks", target.parts)
                self.assertTrue(target.exists(), f"{path.name}: {link}")
                if parts.fragment and target in owned:
                    self.assertIn(unquote(parts.fragment), headings(target.read_text(encoding="utf-8")), link)

    def test_sdk_links_and_unsupported_claims(self):
        for path in DOCS:
            text = path.read_text(encoding="utf-8")
            for link in ("../../docs/python_api.md", "../../docs/python_api_zh.md"):
                self.assertIn(f"]({link})", text)
            for marker in ("spk=true", "segments=[]", "--model auto", "OpenAI API"):
                self.assertIn(marker, text)
            for stale in ("170x realtime", "120x realtime", "3.6x realtime",
                          "Server starts in ~20s", "Works with **any agent framework**",
                          "Fast default with language, emotion, and event tags",
                          "Returns: text, segments (with start/end/speaker), duration"):
                self.assertNotIn(stale, text)

    def test_python_and_shell_examples_are_syntactically_valid(self):
        for path in DOCS:
            text = path.read_text(encoding="utf-8")
            for index, code in enumerate(blocks(text, "python")):
                compile(code, f"{path}:{index}", "exec")
            for index, code in enumerate(blocks(text, "bash")):
                result = subprocess.run(["bash", "-n"], input=code, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, f"{path}:{index}: {result.stderr}")

    def test_local_script_commands_and_options_match_source(self):
        for path in DOCS:
            text = path.read_text(encoding="utf-8")
            if path.name.startswith("README"):
                quick_start = blocks(text, "bash")[0]
                self.assertIn("cd examples/openai_api", quick_start)
                self.assertLess(quick_start.index("cd examples/openai_api"), quick_start.index("python server.py"))
            for code in blocks(text, "bash"):
                for line in code.replace("\\\n", " ").splitlines():
                    args = shlex.split(line, comments=True)
                    if len(args) < 2 or args[0] not in ("python", "bash"):
                        continue
                    script = EXAMPLE / args[1]
                    self.assertTrue(script.is_file(), str(script))
                    if args[0] != "python":
                        continue
                    options = {ast.literal_eval(arg) for node in ast.walk(tree(script))
                               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                               and node.func.attr == "add_argument"
                               for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
                    for arg in args[2:]:
                        if arg.startswith("--"):
                            self.assertIn(arg.split("=", 1)[0], options, str(script))

    def test_form_defaults_and_alias_boundaries_match_source(self):
        example = form_defaults(function(EXAMPLE / "server.py", "transcribe"))
        packaged = form_defaults(function(PACKAGED, "transcribe"))
        self.assertEqual(example["model"], "sensevoice")
        self.assertNotIn("spk", example)
        self.assertEqual(packaged["model"], "fun-asr-nano")
        self.assertIs(packaged["spk"], False)
        for path, name, expected in ((EXAMPLE / "server.py", "MODEL_CONFIGS", True),
                                     (PACKAGED, "FALLBACK_CONFIGS", False)):
            assignment = next(node for node in ast.walk(tree(path))
                              if isinstance(node, ast.Assign) and any(
                                  isinstance(target, ast.Name) and target.id == name for target in node.targets))
            aliases = [ast.literal_eval(key) for key in assignment.value.keys]
            self.assertEqual("paraformer-en" in aliases, expected)

    def test_startup_defaults_match_source(self):
        for path, expected in ((EXAMPLE / "server.py", "sensevoice"),
                               (ROOT / "funasr/bin/server.py", "auto")):
            option = next(node for node in ast.walk(tree(path)) if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                          and node.args and isinstance(node.args[0], ast.Constant)
                          and node.args[0].value == "--model")
            self.assertEqual(next(ast.literal_eval(kw.value) for kw in option.keywords
                                  if kw.arg == "default"), expected)
        startup = function(PACKAGED, "create_app")
        selection = next(node for node in ast.walk(startup) if isinstance(node, ast.IfExp)
                         and isinstance(node.body, ast.Constant) and node.body.value == "fun-asr-nano")
        for device, expected in (("cuda:0", "fun-asr-nano"), ("cpu", "sensevoice"), ("mps", "sensevoice")):
            self.assertEqual(eval(compile(ast.Expression(selection), str(PACKAGED), "eval"), {"device": device}), expected)

    def test_packaged_verbose_example_matches_real_builder(self):
        samples = [json.loads(code) for code in blocks(DOCS[0].read_text(encoding="utf-8"), "json")]
        self.assertEqual(len(samples), 3)
        namespace = {"resolve_transcription_language": lambda requested, result: requested or result["language"]}
        builder = function(PACKAGED, "build_openai_verbose_json")
        exec(compile(ast.Module(body=[builder], type_ignores=[]), str(PACKAGED), "exec"), namespace)
        result = {"text": "recognized speech", "language": "en", "duration": 3.2,
                  "segments": [{"start": 0.0, "end": 3.2, "text": "recognized speech"}]}
        self.assertEqual(namespace[builder.name](result), samples[1])
        self.assertNotIn("speaker", samples[1]["segments"][0])
        self.assertLessEqual(samples[1]["segments"][0]["end"], samples[1]["duration"])

    def test_example_verbose_sample_and_duration_meaning(self):
        samples = [json.loads(code) for code in blocks(DOCS[0].read_text(encoding="utf-8"), "json")]
        handler = function(EXAMPLE / "server.py", "transcribe")
        response = next(node for node in ast.walk(handler) if isinstance(node, ast.Dict)
                        and any(isinstance(key, ast.Constant) and key.value == "duration" for key in node.keys))
        expected = eval(compile(ast.Expression(response), "example-response", "eval"),
                        {"text": "recognized speech", "segments": [], "language": None,
                         "elapsed": 0.42, "model": "sensevoice"})
        self.assertEqual(expected, samples[2])
        source = PACKAGED.read_text(encoding="utf-8")
        self.assertIn("float(sf.info(audio_path).duration)", source)
        self.assertIn('"duration": len(audio_data) / sr', source)
        clients = DOCS[0].read_text(encoding="utf-8")
        self.assertIn("not audio duration", clients)
        self.assertIn("Audio duration in seconds", clients)
        self.assertIn("does not enable diarization", clients)


if __name__ == "__main__":
    unittest.main()
