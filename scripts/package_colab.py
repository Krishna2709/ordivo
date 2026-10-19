"""Create an allowlisted source archive for a managed runtime; excludes .env/outputs."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('src', 'configs', 'notebooks', 'scripts', 'tests', 'docs')
FILES = ('pyproject.toml', 'requirements-cpu.lock', 'requirements-colab-candidate.txt',
         'uv.lock', 'reports/progress.md', 'research/upstream_models.json', 'ARCHITECTURE.md', 'README.md', 'PROJECT_BRIEF.md', 'de01.ipynb')


def source_paths():
    paths = [ROOT/name for name in FILES]
    for name in DIRECTORIES:
        paths.extend((ROOT/name).rglob('*'))
    return [path for path in sorted(set(paths))
            if path.is_file() and not path.is_symlink()
            and not any(p.startswith('.') or p == '__pycache__'
                        for p in path.relative_to(ROOT).parts)
            and path.suffix not in ('.pyc', '.pyo')]


def package(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in source_paths():
            z.write(path,Path('decision-engines')/path.relative_to(ROOT))
    return output


if __name__ == '__main__':
    print(package(ROOT/'dist'/'colab-source.zip'))
