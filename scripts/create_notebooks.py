"""Rebuild thin, output-free notebook wrappers; source functions live in the package."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


runtime_source = (ROOT / 'src/liquid_mcdm/runtime.py').read_text()
setup = runtime_source + '\nimport sys\nprint(require_colab_gpu())\n' + '''REPO = Path(os.environ.get("LIQUID_MCDM_REPO", str(Path.cwd()))).resolve()
if not (REPO / "pyproject.toml").exists() and (REPO.parent / "pyproject.toml").exists():
    REPO = REPO.parent
assert (REPO / "pyproject.toml").exists(), "Upload/extract colab-source.zip or clone the repo, then set LIQUID_MCDM_REPO. See README."
os.chdir(REPO)
print("Project:", REPO)
'''

intro='''All workload execution must use a managed Colab NVIDIA GPU runtime. CPU-only mathematics tests run on that remote VM; do not run them on the local machine.
This notebook assumes the source archive has been extracted or the exact repository revision cloned in the runtime, and the CPU package installed. See README for upload/clone commands. Local VS Code files do not automatically become Colab runtime files. No API key goes into a cell or notebook output.'''

notebooks={
'00_environment_check.ipynb':[
 md('# 00 · Runtime inventory\n\n'+intro),code(setup),
 code('from liquid_mcdm.environment import environment_report\nimport json\nenvironment = environment_report()\nprint(json.dumps(environment, indent=2))'),
 code('import subprocess\nsubprocess.run([sys.executable, "-m", "pip", "check"], check=True)'),
 md('Inventory is not a training compatibility result. High RAM is host memory. Inspect actual GPU and free VRAM; Colab Pro does not reserve an A100. Candidate GPU dependency resolution, loader imports, forward/backward and adapter reload remain required. See `docs/research_findings.md`.')],
'01_mcdm_data_and_baselines.ipynb':[
 md('# 01 · CPU teachers, data and tracking\n\n'+intro),code(setup),
 code('''import yaml
from liquid_mcdm.data import generate_dataset, load_dataset
from liquid_mcdm.evaluation import evaluate_dataset
import json
cfg = yaml.safe_load((REPO / "configs/data.yaml").read_text())
DATA = REPO / cfg["output"]
if not DATA.exists():
    generate_dataset(DATA, count=cfg["count"], seed=cfg["seed"])
else:
    existing = json.loads((DATA / "manifest.json").read_text())
    assert (existing["seed"], existing["count"]) == (cfg["seed"], cfg["count"]), "Existing data differs from config"
    for split in ("train", "val", "test"):
        load_dataset(DATA, split)  # verify persisted hashes and teachers
report = evaluate_dataset(DATA, "test")
REPORT = REPO / "outputs/notebook-baseline.json"
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))'''),
 code('''from liquid_mcdm.tracking import log_cpu_run
tracking = log_cpu_run(DATA / "manifest.json", REPORT, REPO / "outputs/notebook-tracking", mode="offline")
print(tracking)'''),
 md('Exact-teacher self-consistency is a plumbing control, not learned performance. Uniform probabilities use a first-option prediction tie break. Inspect artifacts before `wandb sync`; there is no cloud run link for an offline run.')],
'02_d1_3b_lora_colab.ipynb':[
 md('# 02 · d1-3B LoRA preparation — training shell\n\nNo weights or training are launched by Run All. M3 needs explicit approval after measured GPU compatibility and zero-shot evaluation.\n\n'+intro),code(setup),
 code('''import yaml
import json
from liquid_mcdm.environment import environment_report
cfg = yaml.safe_load((REPO / "configs/d1_3b_lora.yaml").read_text())
print(json.dumps(cfg, indent=2))
print(json.dumps(environment_report(), indent=2))'''),
 md('**Candidate dependencies.** Start a fresh runtime and inspect `pip install --dry-run -r requirements-colab-candidate.txt`. Keep matching Torch/torchvision/CUDA. Only then install, restart the kernel, run `pip check`, record `pip freeze`, and verify model-class imports. This candidate file has not been GPU-tested. The optional install cell is disabled by default.'),
 code('''INSTALL_CANDIDATE_DEPENDENCIES = False
if INSTALL_CANDIDATE_DEPENDENCIES:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-colab-candidate.txt"], check=True)
    print("Restart the kernel, rerun inventory and pip check before continuing.")'''),
 md('**Tokenizer-only preflight:** downloads only pinned tokenizer/config/template files, never model weights. Uses the audited upstream prompt renderer, full-precision values, alias grouping, answer/EOS boundary assertions, and reference prompt masks. This is not execution of the TRL collator.'),
 code('''RUN_TOKENIZER_CHECK = False
if RUN_TOKENIZER_CHECK:
    from liquid_mcdm.training.preflight import tokenizer_preflight
    result = tokenizer_preflight(REPO / "outputs/smoke-v1", REPO / ".cache/d1-tokenizer")
    (REPO / "outputs/tokenizer_preflight.json").write_text(json.dumps(result, indent=2))
    print(result)
else:
    print("Tokenizer preflight is opt-in; historical result (before final edits) is reports/pre_colab_only_snapshot/tokenizer_preflight.json.")'''),
 md('''## Explicit gate before model work

1. Obtain approval to download the pinned base checkpoint and run a bounded GPU compatibility check. Record device, free VRAM, resolved dependencies and cost/time budget.
2. Download to a revision-pinned local snapshot. Standard training loader: `AutoModelForImageTextToText`; custom inference loader: `AutoModel(trust_remote_code=True)`. Local snapshot loading avoids the custom wrapper silently resolving a tokenizer from main.
3. Inspect real `SFTTrainer` collated labels: prompt/pad `-100`; alias and EOS supervised. `completion_only_loss=True`, `packing=False`, no truncation of alternatives. Verify LoRA trainable names and frozen vision tower. Run a bounded forward/backward and standard/custom option-readout parity check.
4. Record zero-shot results on the untouched test split **before** training. Use max over alias/leading-space token groups then restricted-option softmax.
5. Only after these pass, approve the 64-example / 25-step overfit candidate in `configs/d1_3b_lora.yaml`. Use TRL+PEFT, `report_to="wandb"`, explicit offline/online mode and persisted checkpoints. Verify adapter save/reload predictions before a larger run.

The exact upstream [TRL notebook](https://github.com/Liquid4All/cookbook/blob/744c5b628a27adb4f219ff4a1d11e3d70259b369/finetuning/notebooks/sft_for_decision_models_with_trl.ipynb) supplies the model/trainer recipe. This repository intentionally stops at the tested CPU adapter and this preparation shell; an executable GPU trainer is the next M3 increment, after runtime compatibility is measured.

**Resume:** persist checkpoint, config, source revision, dataset manifest and environment freeze in a user-selected Drive folder or approved artifact store. On a fresh runtime recreate the locked environment/base snapshot and pass the verified checkpoint to `trainer.train(resume_from_checkpoint=...)`. Mounting Drive does not itself save local checkpoints; copy them periodically or use the persistent path. Never assume an adapter-only directory includes optimizer/scheduler state.''')],
'03_evaluate_and_compare.ipynb':[
 md('# 03 · Held-out evaluation\n\n'+intro),code(setup),
 code('''from liquid_mcdm.data import load_dataset
from liquid_mcdm.evaluation import baseline_report, evaluate_predictions
import json
TEST = load_dataset(REPO / "outputs/smoke-v1", "test")
print(json.dumps(baseline_report(TEST), indent=2))'''),
 code('''PREDICTIONS = REPO / "outputs/d1-test-probabilities.json"
if PREDICTIONS.exists():
    predictions = json.loads(PREDICTIONS.read_text())
    metrics = evaluate_predictions(TEST, predictions)
    (REPO / "outputs/d1-test-metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)
else:
    print("No d1 predictions exist. No learned-model performance has been measured.")'''),
 md('Prediction schema: `{episode_id: {alternative_id: probability}}`, complete options per eligible episode. Missing predictions fail, rather than silently improving the denominator. Metrics include top-one accuracy, multiclass Brier, ECE and method/family/size/near-tie/weight slices. Full-ranking Kendall/Spearman/NDCG/MRR are intentionally absent from winner-only evaluation. Neural/LightGBM controls belong to later M3/M4 work.')]
}

def write(path,cells):
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    nbf.validate(nb)
    nbf.write(nb,path)


for name,cells in notebooks.items():
    write(ROOT/'notebooks'/name,cells)
# de01.ipynb is maintained independently as the remote bootstrap entry.
