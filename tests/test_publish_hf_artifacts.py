"""Pure manifest checks for the remote Hugging Face publisher."""
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'publish_hf_artifacts', Path(__file__).resolve().parents[1] / 'scripts' / 'publish_hf_artifacts.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def test_file_manifest_is_sorted_and_content_addressed(tmp_path):
    (tmp_path / 'b.txt').write_text('b')
    (tmp_path / 'a.txt').write_text('a')
    manifest = publisher.file_manifest(tmp_path)
    assert [item['path'] for item in manifest] == ['a.txt', 'b.txt']
    assert manifest[0]['bytes'] == 1
    assert manifest[0]['sha256'] == publisher.sha256(tmp_path / 'a.txt')
