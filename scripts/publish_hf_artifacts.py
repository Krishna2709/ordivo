"""Publish a completed remote run directly to Hugging Face without local weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(root):
    root = Path(root)
    return [
        {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
         'sha256': sha256(path)}
        for path in sorted(root.rglob('*')) if path.is_file()
    ]


def token_from_environment():
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_API_KEY')
    if not token:
        raise RuntimeError('HF_TOKEN or HUGGINGFACE_API_KEY is required')
    return token


def publish(data_dir, training_dir, source_root, model_repo, dataset_repo, receipt_path):
    from huggingface_hub import HfApi, hf_hub_download

    data_dir = Path(data_dir).resolve()
    training_dir = Path(training_dir).resolve()
    source_root = Path(source_root).resolve()
    state = json.loads((training_dir / 'run.json').read_text())
    if not state.get('training_started') or not (training_dir / 'adapter').is_dir():
        raise RuntimeError('A completed adapter directory and training receipt are required')

    token = token_from_environment()
    api = HfApi(token=token)
    identity = api.whoami()['name']
    for repo in (model_repo, dataset_repo):
        if repo.split('/', 1)[0] != identity:
            raise RuntimeError('Publish target is outside the authenticated user namespace')

    api.create_repo(dataset_repo, repo_type='dataset', private=False, exist_ok=True)
    api.create_repo(model_repo, repo_type='model', private=False, exist_ok=True)
    api.upload_folder(repo_id=dataset_repo, repo_type='dataset', folder_path=data_dir,
                      commit_message='Publish deterministic MCDM dataset')
    api.upload_file(repo_id=dataset_repo, repo_type='dataset',
                    path_or_fileobj=source_root / 'hub' / 'dataset' / 'README.md',
                    path_in_repo='README.md', commit_message='Update dataset card')

    api.upload_folder(repo_id=model_repo, repo_type='model',
                      folder_path=training_dir / 'adapter',
                      commit_message='Publish d1-3B MCDM LoRA adapter')
    api.upload_file(repo_id=model_repo, repo_type='model',
                    path_or_fileobj=source_root / 'hub' / 'model' / 'README.md',
                    path_in_repo='README.md', commit_message='Update model card')
    evidence_names = [
        'run.json', 'base_metrics.json', 'post_metrics.json',
        'base_train_metrics.json', 'post_train_metrics.json',
        'comparison_base_metrics.json', 'comparison_post_metrics.json',
        'prediction_rows.json', 'custom_standard_parity.json',
        'adapter_reload_parity.json', 'adapter_repeat_inference_parity.json',
        'adapter_fresh_reload_parity.json', 'pip-freeze.txt',
    ]
    for name in evidence_names:
        path = training_dir / name
        if path.is_file():
            api.upload_file(repo_id=model_repo, repo_type='model', path_or_fileobj=path,
                            path_in_repo=f'evidence/{name}',
                            commit_message=f'Add run evidence {name}')

    expected_data = file_manifest(data_dir)
    expected_adapter = file_manifest(training_dir / 'adapter')
    with tempfile.TemporaryDirectory(dir=training_dir) as temp:
        downloaded_data = []
        for item in expected_data:
            local = hf_hub_download(dataset_repo, item['path'], repo_type='dataset',
                                    token=token, local_dir=temp, force_download=True)
            downloaded_data.append({'path': item['path'], 'bytes': Path(local).stat().st_size,
                                    'sha256': sha256(local)})
    remote_model_files = set(api.list_repo_files(model_repo, repo_type='model'))
    remote_dataset_files = set(api.list_repo_files(dataset_repo, repo_type='dataset'))
    adapter_paths = {item['path'] for item in expected_adapter}
    if downloaded_data != expected_data:
        raise RuntimeError('Dataset read-back differs from uploaded files')
    if not adapter_paths.issubset(remote_model_files):
        raise RuntimeError('Model repository is missing adapter files')
    if not {item['path'] for item in expected_data}.issubset(remote_dataset_files):
        raise RuntimeError('Dataset repository is missing files')

    remote_model_sizes = {item.path: item.size for item in
                          api.list_repo_tree(model_repo, repo_type='model', recursive=True)
                          if hasattr(item, 'size')}
    if any(remote_model_sizes.get(item['path']) != item['bytes'] for item in expected_adapter):
        raise RuntimeError('Model repository adapter sizes differ from local files')

    receipt = {
        'authenticated_user': identity,
        'model_repo': model_repo,
        'dataset_repo': dataset_repo,
        'model_url': f'https://huggingface.co/{model_repo}',
        'dataset_url': f'https://huggingface.co/datasets/{dataset_repo}',
        'dataset_files': expected_data,
        'adapter_files': expected_adapter,
        'dataset_readback_verified': True,
        'model_file_presence_verified': True,
        'model_file_sizes_verified': True,
    }
    Path(receipt_path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: receipt[key] for key in
                      ('model_url', 'dataset_url', 'dataset_readback_verified',
                       'model_file_presence_verified')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--model-repo', default='blackOrder/d1-3b-mcdm-lora')
    parser.add_argument('--dataset-repo', default='blackOrder/liquid-mcdm-synthetic')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    publish(args.data, args.training, args.source_root, args.model_repo,
            args.dataset_repo, args.receipt)


if __name__ == '__main__':
    main()
