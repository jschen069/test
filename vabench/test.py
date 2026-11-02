vabench_eval.py

"""AISBench eval-only VABench task with an isolated official backend."""
import argparse
import os.path as osp
import shlex
import subprocess
import sys
import threading

from mmengine.config import Config

from ais_bench.benchmark.registry import TASKS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.utils.core.abbr import (get_infer_output_path,
                                               task_abbr_from_cfg)
from ais_bench.benchmark.vabench.engine import run, successful
from ais_bench.benchmark.vabench.data import read_json, write_json


def _case_progress(sample_count, done, total):
    """Map completed metric groups onto a case-count progress bar."""
    if not sample_count or not total:
        return 0
    return min(sample_count, round(sample_count * done / total))


@TASKS.register_module()
class VABenchEvalTask(BaseTask):
    name_prefix = 'VABenchEval'
    log_subdir = 'logs/eval'
    output_subdir = 'results'

    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_gpus = int(cfg.get('eval', {}).get('runner', {}).get('task', {}).get('num_gpus', 1))

    def get_command(self, cfg_path, template):
        command = [sys.executable, '-m', 'ais_bench.benchmark.tasks.vabench_eval', str(cfg_path)]
        rendered = subprocess.list2cmdline(command) if sys.platform == 'win32' else shlex.join(command)
        return template.format(task_cmd=rendered)

    def run(self, task_state_manager=None):
        failed = []
        for dataset in self.dataset_cfgs:
            options = dict(dataset.get('eval_cfg', {}))
            source = options.pop('source_root')
            cache = options.pop('cache_dir')
            prepared = dataset['path']
            destination = get_infer_output_path(
                self.model_cfg, dataset, osp.join(self.work_dir, self.output_subdir))
            sample_count = len(read_json(osp.join(prepared, 'manifest.json'))['records'])
            if task_state_manager:
                task_state_manager.update_task_state(dict(
                    status='evaluating', total_count=sample_count, finish_count=0,
                    progress_description='', other_kwargs={'dimension': 'starting'}))

            def _on_progress(done, total, dimensions):
                if task_state_manager:
                    task_state_manager.update_task_state(dict(
                        status='evaluating', total_count=sample_count,
                        finish_count=_case_progress(sample_count, done, total),
                        progress_description='',
                        other_kwargs={'dimension': ','.join(dimensions)}))

            report = run(prepared, source, cache, destination[:-5] + '_artifacts',
                         progress=_on_progress if task_state_manager else None, **options)
            write_json(destination, report)
            if not successful(report):
                failed.append(dataset.get('abbr', prepared))
            if task_state_manager:
                task_state_manager.update_task_state(dict(
                    status='evaluating', total_count=sample_count, finish_count=sample_count,
                    progress_description='', other_kwargs={'dimension': 'done'}))
        if failed:
            raise RuntimeError(f'VABench incomplete; reports preserved: {failed}')

    def get_output_paths(self, file_extension='json'):
        return [get_infer_output_path(self.model_cfg, dataset,
                                     osp.join(self.work_dir, self.output_subdir), file_extension)
                for dataset in self.dataset_cfgs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    manager = TaskStateManager(tmp_path=osp.join(cfg['work_dir'], 'status_tmp'),
                               task_name=task_abbr_from_cfg(cfg),
                               is_debug=cfg.get('cli_args', {}).get('debug', False))
    monitor = threading.Thread(target=manager.launch, daemon=True)
    monitor.start()
    try:
        manager.update_task_state(dict(status='start', task_log_path=osp.join(
            'logs/eval', f'{task_abbr_from_cfg(cfg)}.out')))
        VABenchEvalTask(cfg).run(manager)
    except Exception:
        manager.update_task_state({'status': 'error'})
        raise
    else:
        manager.update_task_state({'status': 'finish'})
    finally:
        monitor.join(timeout=10)


if __name__ == '__main__':
    main()

	
cli.py

"""Lightweight preparation, runtime diagnostics, evaluation and comparison.

Usage: python3 -m ais_bench.benchmark.vabench.cli --help
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from . import DIMENSIONS
from .data import (PROTOCOL_PATH, catalog, catalog_videos, digest, media_tool,
                   prepare, read_json, write_json)
from .engine import run, successful
from .resources import preflight
from .results import (compare, export_csv, export_official_results, format_summary,
                      refresh_report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('catalog', 'prepare', 'doctor', 'run'):
        cmd = commands.add_parser(command)
        cmd.add_argument('--source-root', required=True)
        if command == 'catalog':
            cmd.add_argument('--subset', choices=('standard', 'mini', 'smoke'), default='mini')
            cmd.add_argument('--sample-id', help='Exact pinned idx; required only for --subset smoke')
        if command == 'prepare':
            cmd.add_argument('--subset', choices=('standard', 'mini', 'smoke', 'custom'), default='mini')
            cmd.add_argument('--sample-id', help='Exact pinned idx; required only for --subset smoke')
        if command in ('catalog', 'prepare', 'run'):
            cmd.add_argument('--output', required=True)
        if command == 'prepare':
            cmd.add_argument('--videos', required=True)
        if command in ('doctor', 'run'):
            cmd.add_argument('--cache-dir', required=True)
            cmd.add_argument('--dimensions', nargs='+', choices=DIMENSIONS, default=list(DIMENSIONS))
        if command == 'run':
            cmd.add_argument('--prepared', required=True)
            cmd.add_argument('--device', default='cuda:0')
            cmd.add_argument('--python', default=sys.executable)
            cmd.add_argument('--mode', choices=('adapter', 'official'), default='adapter')
            cmd.add_argument('--options', help='JSON: params, dim_params, device_overrides, timeout')
    cmp = commands.add_parser('compare')
    cmp.add_argument('--reference', required=True)
    cmp.add_argument('--candidate', required=True)
    cmp.add_argument('--output', required=True)
    cmp.add_argument('--dimensions', nargs='+', choices=DIMENSIONS, default=list(DIMENSIONS))
    cmp.add_argument('--relative', type=float, default=0.01)
    cmp.add_argument('--near-zero', type=float, default=1e-6)
    cmp.add_argument('--absolute', type=float, default=1e-6)
    summary = commands.add_parser('summarize', help='Reaggregate an AISBench per-sample report without model inference')
    summary.add_argument('--report', required=True)
    summary.add_argument('--output', required=True, help='New output directory; original report is retained')
    summary.add_argument('--prepared', help='Prepared directory used to restore official prompt names')
    args = parser.parse_args(argv)
    if args.command == 'catalog':
        entries = catalog(args.source_root, args.subset, sample_id=args.sample_id)
        protocol = read_json(PROTOCOL_PATH)
        write_json(args.output, dict(subset=args.subset, count=len(entries),
                                    upstream_commit=protocol['commit'], entries=entries,
                                    mini_selection=protocol['mini'] if args.subset == 'mini' else []))
        print(f'Wrote {len(entries)} prompts to {args.output}')
    elif args.command == 'prepare':
        if args.subset == 'custom':
            if args.sample_id:
                parser.error('--sample-id is only valid with --subset smoke')
            entries = catalog_videos(args.source_root, args.videos)
        else:
            entries = catalog(args.source_root, args.subset, sample_id=args.sample_id)
        result = prepare(entries, args.videos, args.output, subset=args.subset)
        print(f'Prepared {result["subset_size"]} samples at {args.output}')
        if args.subset == 'custom':
            print('Custom subset of the pinned 778; reports are not official Mini10/778 results.')
    elif args.command == 'doctor':
        info = dict(python=sys.version, executable=sys.executable,
                    platform=platform.platform(),
                    ffmpeg=shutil.which(media_tool('ffmpeg')), ffprobe=shutil.which(media_tool('ffprobe')),
                    modules={m: importlib.util.find_spec(m) is not None for m in (
                        'torch', 'torch_npu', 'transformers', 'qwen_omni_utils',
                        'nisqa', 'audiobox_aesthetics', 'onnxruntime', 'decord', 'imagebind')},
                    missing_resources=preflight(args.source_root, args.cache_dir, args.dimensions))
        try:
            import torch
            info['torch'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            if info['modules']['torch_npu']:
                import torch_npu  # noqa: F401
                info['npu_available'] = torch.npu.is_available()
                info['npu_count'] = torch.npu.device_count()
        except Exception as exc:
            info['device_error'] = str(exc)
        if shutil.which('npu-smi'):
            result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True)
            info['npu_smi'] = result.stdout or result.stderr
        print(json.dumps(info, ensure_ascii=False, indent=2))
        required = {'torch'}
        by_dimension = {'first_dnsmos': ['onnxruntime'], 'first_nisqa': ['nisqa'],
                        'first_audiobox': ['audiobox_aesthetics'],
                        'second_viclip': ['decord'], 'second_imagebind': ['imagebind'],
                        'second_lsa': ['onnxruntime']}
        for dimension in args.dimensions:
            required.update(by_dimension.get(dimension, []))
            if dimension.startswith(('third_', 'fourth_')):
                required.update(['transformers', 'qwen_omni_utils'])
        return int(bool(info['missing_resources']) or not info['ffmpeg'] or not info['ffprobe']
                   or any(not info['modules'][name] for name in required))
    elif args.command == 'run':
        options = read_json(args.options) if args.options else {}
        allowed = {'params', 'dim_params', 'device_overrides', 'timeout'}
        if set(options) - allowed:
            parser.error(f'Unknown options: {set(options) - allowed}')
        result = run(args.prepared, args.source_root, args.cache_dir, args.output,
                     args.dimensions, device=args.device, python=args.python, mode=args.mode, **options)
        print(format_summary(result))
        print(f'Report: {Path(args.output).resolve() / "report.json"}')
        return int(not successful(result))
    elif args.command == 'summarize':
        destination = Path(args.output).resolve()
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f'Choose a new summary directory: {destination}')
        result = refresh_report(read_json(args.report))
        result['postprocessing'] = dict(source_report=str(Path(args.report).resolve()),
                                        source_report_sha256=digest(args.report), model_inference_rerun=False)
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / 'report.json', result)
        export_csv(result, destination)
        export_official_results(result, destination, prepared=args.prepared)
        print(format_summary(result))
        print(f'Report: {destination / "report.json"}')
        return 0
    else:
        result = compare(read_json(args.reference), read_json(args.candidate),
                         args.dimensions, args.relative, args.near_zero, args.absolute)
        write_json(args.output, result)
        print('PASS' if result['passed'] else 'FAIL')
        return int(not result['passed'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

	
data.py

"""Pinned T2AV protocol and lossless media preparation (stdlib only)."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


PROTOCOL_PATH = Path(__file__).with_name('protocol.json')


def media_tool(name):
    return os.environ.get(f'VABENCH_{name.upper()}', name)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    """Replace atomically: interrupted work must not look complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def object_digest(data):
    return hashlib.sha256(json.dumps(
        data, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def text_digest(path):
    return hashlib.sha256(Path(path).read_text(
        encoding='utf-8').encode('utf-8')).hexdigest()


def validate_idx(idx):
    if (not idx or idx in ('.', '..') or '/' in idx or '\\' in idx
            or Path(idx).is_absolute()):
        raise ValueError(f'Unsafe official idx: {idx!r}')


def catalog(source_root, subset='standard', sample_id=None):
    """Read exact upstream prompts, never repair or fuzzy-match them."""
    root = Path(source_root).resolve()
    protocol = read_json(PROTOCOL_PATH)
    if subset not in ('standard', 'mini', 'smoke'):
        raise ValueError('subset must be standard, mini or smoke')
    if subset == 'smoke':
        if sample_id not in protocol['standard_ids']:
            raise ValueError('smoke requires one sample_id from the pinned 778 IDs')
    elif sample_id is not None:
        raise ValueError('sample_id is only supported with subset=smoke')
    changed = [p for p, sha in protocol['data_sha256'].items()
               if not (root / p).is_file() or text_digest(root / p) != sha]
    if changed:
        raise ValueError(f'Official data differs from pinned protocol: {changed}')
    mapping = {}
    ids = set()
    with (root / 'mapping/final_idx_to_prompt.csv').open(
            encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            prompt = json.loads('"' + row['prompt'] + '"')
            idx = row['idx']
            validate_idx(idx)
            if idx in ids or prompt in mapping:
                raise ValueError('Duplicate idx/prompt in official mapping')
            ids.add(idx)
            mapping[prompt] = idx
    selected = ({sample_id} if subset == 'smoke' else
                set(protocol['standard_ids'] if subset == 'standard'
                    else [x['idx'] for x in protocol['mini']]))
    entries = []
    for file in sorted((root / 'json_files/json_text').glob('*.json')):
        for item in read_json(file):
            idx = mapping.get(item['prompt_en'])
            if idx in selected:
                entries.append(dict(idx=idx, category=file.stem, metadata=item))
    if len(entries) != len(selected) or {e['idx'] for e in entries} != selected:
        raise ValueError('Pinned subset does not match JSON/mapping')
    return entries


def catalog_videos(source_root, videos_path):
    """Pinned entries for every official MP4 stem found under videos_path.

    Custom runs stay inside the pinned 778 prompts because the official
    metrics need each sample's JSON prompt and auxiliary metadata. Unknown
    videos are rejected instead of being skipped silently.
    """
    entries = catalog(source_root, 'standard')
    by_idx = {entry['idx']: entry for entry in entries}
    videos = Path(videos_path).resolve()
    if not videos.is_dir():
        raise FileNotFoundError(videos)
    by_stem = {}
    for path in videos.rglob('*'):
        if path.is_file() and path.suffix.lower() == '.mp4':
            by_stem.setdefault(path.stem, []).append(path)
    unknown = sorted(set(by_stem) - set(by_idx))
    if unknown:
        raise ValueError(f'Videos outside the pinned 778 catalog: {unknown}')
    ambiguous = sorted(idx for idx, paths in by_stem.items() if len(paths) > 1)
    if ambiguous:
        raise ValueError(f'Ambiguous MP4 stems: {ambiguous}')
    selected = [by_idx[idx] for idx in sorted(by_stem)]
    if not selected:
        raise ValueError('No pinned T2AV videos found in the given directory')
    return selected


def probe(path):
    result = subprocess.run([
        media_tool('ffprobe'), '-v', 'error', '-show_streams', '-show_format',
        '-of', 'json', str(path)], check=True, capture_output=True,
        text=True, encoding='utf-8')
    info = json.loads(result.stdout)
    video = [s for s in info['streams'] if s['codec_type'] == 'video']
    audio = [s for s in info['streams'] if s['codec_type'] == 'audio']
    if not video or not audio:
        raise ValueError(f'MP4 must contain video and audio: {path}')
    duration = float(info.get('format', {}).get('duration', 0))
    if duration <= 0:
        raise ValueError(f'Invalid media duration: {path}')
    return dict(duration=duration, video=video[0], audio=audio[0])


def prepare(entries, videos_path, output_path, subset=None):
    """Strict input set, original MP4 + unresampled PCM audio for experts.

    Missing/ambiguous samples fail preparation, rather than shrinking a mean.
    Subsets are prepared once before parallel evaluation tasks.
    """
    if subset == 'smoke' and len(entries) != 1:
        raise ValueError('smoke preparation requires exactly one sample')
    videos = Path(videos_path).resolve()
    output = Path(output_path).resolve()
    if not videos.is_dir():
        raise FileNotFoundError(videos)
    if (output / 'manifest.json').exists():
        raise FileExistsError(f'Choose a new preparation directory: {output}')
    by_stem = {}
    for path in videos.rglob('*'):
        if path.is_file() and path.suffix.lower() == '.mp4':
            by_stem.setdefault(path.stem, []).append(path)
    missing = [e['idx'] for e in entries if not by_stem.get(e['idx'])]
    duplicate = [e['idx'] for e in entries if len(by_stem.get(e['idx'], [])) > 1]
    if missing or duplicate:
        raise ValueError(f'Missing MP4 ({len(missing)}): {missing}; '
                         f'ambiguous MP4 ({len(duplicate)}): {duplicate}')
    grouped, records = {}, []
    for entry in entries:
        idx, category = entry['idx'], entry['category']
        validate_idx(idx)
        path = by_stem[idx][0]
        info = probe(path)
        wav = output / 'audio' / category / (idx + '.wav')
        wav.parent.mkdir(parents=True, exist_ok=True)
        # No trim, channel mixing, resampling, or video re-encode.
        # The official individual metrics perform their own preprocessing.
        subprocess.run([
            media_tool('ffmpeg'), '-nostdin', '-v', 'error', '-y', '-i', str(path),
            '-map', '0:a:0', '-vn', '-c:a', 'pcm_s16le', str(wav)], check=True)
        item = dict(entry['metadata'], video_list=[str(path)], audio_list=[str(wav)])
        grouped.setdefault(category, []).append(item)
        records.append(dict(idx=idx, category=category, video_path=str(path),
                            audio_path=str(wav), video_sha256=digest(path),
                            audio_sha256=digest(wav), duration=info['duration'],
                            short_video=info['duration'] < 5))
    for category, items in grouped.items():
        write_json(output / 'json' / f'{category}.json', items)
    manifest = dict(schema_version=1, subset_size=len(entries),
                    evaluation_scope=subset or 'custom',
                    protocol_sha256=digest(PROTOCOL_PATH), records=records,
                    ffmpeg=subprocess.check_output(
                        [media_tool('ffmpeg'), '-version'], text=True,
                        encoding='utf-8').splitlines()[0])
    manifest['input_signature'] = input_signature(records)
    manifest['metadata_signature'] = object_digest(grouped)
    write_json(output / 'manifest.json', manifest)
    return manifest


def input_signature(records):
    return object_digest(sorted([
        {k: r[k] for k in ('idx', 'category', 'video_sha256', 'audio_sha256')}
        for r in records], key=lambda r: r['idx']))


def verify_prepared(prepared):
    """Recheck immutable inputs before each run (including overwritten WAV)."""
    root = Path(prepared).resolve()
    manifest = read_json(root / 'manifest.json')
    if manifest['protocol_sha256'] != digest(PROTOCOL_PATH):
        raise ValueError('Prepared protocol differs from installed protocol')
    records = manifest['records']
    if len({r['idx'] for r in records}) != len(records):
        raise ValueError('Duplicate samples in preparation manifest')
    for r in records:
        for media in ('video', 'audio'):
            if digest(r[f'{media}_path']) != r[f'{media}_sha256']:
                raise ValueError(f'Input changed after preparation: {r["idx"]} {media}')
    grouped = {p.stem: read_json(p) for p in sorted((root / 'json').glob('*.json'))}
    if object_digest(grouped) != manifest['metadata_signature']:
        raise ValueError('Prepared metadata changed')
    if input_signature(records) != manifest['input_signature']:
        raise ValueError('Input manifest signature mismatch')
    return manifest, grouped

	
engine.py

"""Run official algorithms in isolated, device-specific subprocesses."""
import os
from pathlib import Path
import subprocess
import sys

from . import DIMENSIONS, MLLM_DIMENSIONS
from .data import (PROTOCOL_PATH, catalog, digest, object_digest, read_json,
                   verify_prepared, write_json)
from .resources import preflight, source_signature, weight_fingerprints
from .results import (build_report, export_csv, export_official_results,
                      normalize_records)


def _validate_prepared_split(manifest, source_root):
    """Official splits stay exact; custom is an explicit non-empty 778 subset."""
    standard = {r['idx'] for r in catalog(source_root, 'standard')}
    actual = {r['idx'] for r in manifest['records']}
    scope = manifest.get('evaluation_scope')
    if scope == 'smoke':
        if len(actual) != 1 or not actual <= standard:
            raise ValueError('Smoke run requires exactly one pinned T2AV sample')
        return
    if scope == 'custom':
        if not actual:
            raise ValueError('Custom run requires at least one sample')
        if not actual <= standard:
            raise ValueError('Custom run contains samples outside the pinned 778 catalog')
        return
    mini = {r['idx'] for r in catalog(source_root, 'mini')}
    if actual != standard and actual != mini:
        raise ValueError('Prepared sample set must be the pinned 778/10 split, '
                         'an explicit custom subset, or a single-sample smoke run')


def run(prepared, source_root, cache_dir, output, dimensions=None, device='cuda:0',
        python=None, device_overrides=None, params=None, dim_params=None,
        mode='adapter', timeout=86400, progress=None):
    dimensions = list(DIMENSIONS if dimensions is None else dimensions)
    if not dimensions or len(set(dimensions)) != len(dimensions) or set(dimensions) - set(DIMENSIONS):
        raise ValueError('Unknown, duplicate, or empty VABench dimensions')
    if mode not in ('adapter', 'official'):
        raise ValueError('mode must be adapter or official')
    overrides = dict(device_overrides or {})
    if set(overrides) - set(DIMENSIONS):
        raise ValueError('Unknown device override')
    if any(d in MLLM_DIMENSIONS for d in overrides):
        raise ValueError('MLLM dimensions share a single device; use device=')
    manifest, grouped = verify_prepared(prepared)
    _validate_prepared_split(manifest, source_root)
    output = Path(output).resolve()
    if (output / 'report.json').exists():
        raise FileExistsError(f'Choose a new run directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'input_manifest.json', manifest)
    # DNSMOS official ONNX scorer uses CPU. Declare it, including on CUDA runs.
    overrides.setdefault('first_dnsmos', 'cpu')
    jobs = [[d] for d in dimensions if d not in MLLM_DIMENSIONS]
    mllm = [d for d in dimensions if d in MLLM_DIMENSIONS]
    if mllm:
        jobs.append(mllm)
    print('Hashing local weights for provenance (may take several minutes).', flush=True)
    hashes, missing_weights = weight_fingerprints(dimensions, cache_dir)
    provenance = dict(protocol_sha256=digest(PROTOCOL_PATH),
                      source_signature=source_signature(source_root),
                      integration_signature=object_digest({p.name: digest(p) for p in
                          sorted(Path(__file__).parent.glob('*.py'))}),
                      weights_signature=object_digest(hashes) if not missing_weights else None,
                      scoring_signature=object_digest(dict(params=params or {}, dim_params=dim_params or {})),
                      weights=hashes, missing_weights=missing_weights,
                      mode=mode, device=device, device_overrides=overrides,
                      scoring_params=params or {}, dim_params=dim_params or {}, environments=[])
    results = {}
    for number, dims in enumerate(jobs):
        work = output / f'job_{number:02d}'
        work.mkdir(parents=True, exist_ok=True)
        request = dict(prepared=str(Path(prepared).resolve()),
                       source_root=str(Path(source_root).resolve()),
                       cache_dir=str(Path(cache_dir).resolve()),
                       device=overrides.get(dims[0], device), dimensions=dims,
                       work_dir=str(work), response=str(work / 'response.json'),
                       records=manifest['records'], grouped=grouped,
                       params=params or {}, dim_params=dim_params or {}, mode=mode)
        write_json(work / 'request.json', request)
        missing = preflight(source_root, cache_dir, dims)
        error = None
        response = {}
        if missing:
            error = 'Missing local resources: ' + ', '.join(missing)
        else:
            env = dict(os.environ)
            package_root = str(Path(__file__).resolve().parents[3])
            env['PYTHONPATH'] = package_root + os.pathsep + env.get('PYTHONPATH', '')
            env['PYTHONUTF8'] = '1'
            env['PYTHONHASHSEED'] = '42'
            command = [python or sys.executable, '-m',
                       'ais_bench.benchmark.vabench.worker', str(work / 'request.json')]
            print(f'Running {dims} on {request["device"]}', flush=True)
            try:
                with (work / 'process.log').open('w', encoding='utf-8') as log:
                    proc = subprocess.run(command, env=env, cwd=work, stdout=log,
                                          stderr=subprocess.STDOUT, timeout=timeout)
                response = read_json(work / 'response.json')
                error = response.get('error')
                if proc.returncode and not error:
                    error = f'Worker exited {proc.returncode}; inspect process.log'
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        if error:
            for dim in dims:
                results[dim] = dict(records=normalize_records(
                    dim, [], manifest['records'], backend_errors=True),
                    errors=[dict(error=error)], requested_device=request['device'],
                    device_observations=[])
        else:
            for dim in dims:
                results[dim] = response['details'][dim]
            provenance['environments'].append(response.get('environment', {}))
        report = build_report(manifest, results, provenance, requested_dimensions=dimensions)
        write_json(output / 'partial_report.json', report)
        if progress is not None:
            # Progress reporting must never affect scoring.
            try:
                progress(number + 1, len(jobs), list(dims))
            except Exception:
                pass
    report = build_report(manifest, results, provenance, requested_dimensions=dimensions)
    write_json(output / 'report.json', report)
    export_csv(report, output)
    export_official_results(report, output, prepared=Path(prepared).resolve())
    return report


def successful(report):
    # A selected single-metric smoke run may succeed without a 15-metric total.
    # LSA omissions nevertheless make the selected evaluation incomplete.
    selected = report.get('requested_dimensions', list(report.get('details', {})) or list(report['metrics']))
    return bool(selected) and all(
        (m := report['metrics'].get(d, {})).get('scored', 0) > 0
        and not m.get('failed') and not m.get('missing') and not m.get('not_evaluated')
        and m.get('status', 'complete') == 'complete'
        and m.get('expected', m['scored']) == m['scored'] for d in selected)

		
result.py

"""Official raw scores plus fixed display scores, coverage and an AISBench total."""
import csv
from datetime import datetime
import os
from pathlib import Path
from statistics import mean

from . import DIMENSIONS
from .data import object_digest, read_json, write_json
from .scoring import RULES, aggregate, finite_number, normalize_score, protocol


def record_failure(dimension, score, raw):
    raw = raw or {}
    reasons = [str(raw.get('reasoning', ''))]
    answers = raw.get('answers')
    if isinstance(answers, list):
        reasons.extend(str(a.get('reason', '')) for a in answers if isinstance(a, dict))
    if any('OOM_ERROR' in reason.upper() for reason in reasons):
        return 'backend_error'
    if not finite_number(score):
        return 'invalid_score'
    if dimension.startswith('third_'):
        if any('empty or oom/parse fallback' in reason.lower() for reason in reasons):
            return 'parse_error'
        if not 1 <= score <= 5:
            return 'invalid_score'
    if dimension.startswith('fourth_'):
        if not 0 <= score <= 1:
            return 'invalid_score'
        if answers is not None and (not isinstance(answers, list) or not answers or any(
                not isinstance(a, dict) or a.get('answer') not in ('yes', 'no')
                or not str(a.get('reason', '')).strip() for a in answers)):
            return 'parse_error'
    return None


def normalize_records(dimension, raw, expected, backend_errors=False, failed_ids=None):
    """Keep missing/failed outputs distinct from legitimate numerical zero.

    ``failed_ids`` marks samples the backend named explicitly, so one bad
    clip in a scene does not invalidate valid scores from its neighbours.
    """
    failed_ids = set(failed_ids or ())
    lookup = {}
    for row in raw:
        path = row.get('video_path') or row.get('audio_path')
        if not path:
            raise ValueError(f'{dimension}: result without media path')
        idx = Path(path).stem
        if idx in lookup:
            raise ValueError(f'{dimension}: duplicate result {idx}')
        lookup[idx] = row
    if set(lookup) - {e['idx'] for e in expected}:
        raise ValueError(f'{dimension}: unexpected result sample')
    result = []
    for entry in expected:
        row = lookup.get(entry['idx'])
        score = row.get('video_results') if row else None
        status = ('backend_error' if backend_errors or entry['idx'] in failed_ids
                  else 'missing' if row is None
                  else record_failure(dimension, score, row) or 'ok')
        result.append(dict(idx=entry['idx'], category=entry['category'], status=status,
                           score=score if status == 'ok' else None,
                           original_score=score if finite_number(score) else None, raw=row))
    return result


def summarize_dimension(records, dimension=None):
    def stats(rows):
        valid = [r['score'] for r in rows if r['status'] == 'ok' and finite_number(r.get('score'))]
        originals = [r.get('original_score', (r.get('raw') or {}).get('video_results', r.get('score')))
                     for r in rows]
        originals = [v for v in originals if finite_number(v)]
        expected = len(rows)
        report = dict(expected=expected, scored=len(valid),
                      missing=sum(r['status'] == 'missing' for r in rows),
                      failed=sum(r['status'] not in ('ok', 'missing', 'not_evaluated') for r in rows),
                      not_evaluated=sum(r['status'] == 'not_evaluated' for r in rows),
                      mean=mean(valid) if valid else None,
                      official_numeric_mean=mean(originals) if originals else None,
                      coverage=len(valid) / expected if expected else 0.0,
                      coverage_percent=100 * len(valid) / expected if expected else 0.0)
        report['omitted'] = report['missing'] if dimension == 'second_lsa' else 0
        report['status'] = ('complete' if expected and len(valid) == expected else
                            'not_evaluated' if report['not_evaluated'] == expected else 'incomplete')
        if dimension is not None:
            report.update(normalize_score(dimension, report['mean']))
            low, high, _ = RULES[dimension]
            report['out_of_bounds_count'] = sum(v < low or v > high for v in valid)
        return report
    report = stats(records)
    report['scenes'] = {cat: stats([r for r in records if r['category'] == cat])
                        for cat in sorted({r['category'] for r in records})}
    return report


def build_report(manifest, dimensions, provenance, requested_dimensions=None):
    scope = manifest.get('evaluation_scope', 'legacy')
    expected = {r['idx']: r for r in manifest['records']}
    if len(expected) != len(manifest['records']):
        raise ValueError('Duplicate input sample IDs')
    details, metrics = {}, {}
    for dimension in DIMENSIONS:
        value = dimensions.get(dimension)
        rows = value.get('records', []) if value is not None else []
        lookup = {r['idx']: r for r in rows}
        if len(lookup) != len(rows) or set(lookup) - set(expected):
            raise ValueError(f'{dimension}: duplicate/unexpected sample records')
        aligned = []
        for idx, entry in expected.items():
            if idx not in lookup:
                aligned.append(dict(idx=idx, category=entry.get('category', 'unknown'), score=None,
                                    original_score=None, raw=None,
                                    status='not_evaluated' if value is None else 'missing'))
                continue
            row = dict(lookup[idx])
            row.setdefault('category', entry.get('category', 'unknown'))
            row.setdefault('original_score', (row.get('raw') or {}).get('video_results', row.get('score')))
            if row['status'] == 'ok':
                row['status'] = record_failure(dimension, row.get('score'), row.get('raw')) or 'ok'
            if row['status'] != 'ok':
                row['score'] = None
            aligned.append(row)
        metrics[dimension] = summarize_dimension(aligned, dimension)
        if value is not None:
            metrics[dimension]['backend_issue_count'] = len(value.get('errors', []))
            if value.get('errors') and metrics[dimension]['status'] == 'complete':
                metrics[dimension]['status'] = 'incomplete'
            details[dimension] = dict(value, records=aligned)
    selected = list(dimensions) if requested_dimensions is None else list(requested_dimensions)
    normalization = protocol()
    scope_note = ''
    if scope == 'smoke':
        scope_note = 'Single-sample smoke test; not Mini10/778 benchmark acceptance.'
    elif scope == 'custom':
        scope_note = (f'Custom subset of {len(expected)} pinned sample(s); '
                      'not Mini10/778 benchmark acceptance.')
    total = aggregate(metrics)
    if scope == 'custom' and total['status'] in ('formal', 'preview'):
        total['label'] = ('自定义子集总分（非官方）' if total['status'] == 'formal'
                          else '自定义子集预览总分（非官方）')
    return dict(schema_version=2, input_signature=manifest['input_signature'], evaluation_scope=scope,
                scope_note=scope_note,
                sample_ids=sorted(expected), requested_dimensions=selected,
                provenance=provenance, details=details, metrics=metrics,
                directions={d: 'lower' if d == 'second_desync' else 'higher' for d in DIMENSIONS},
                normalization_protocol=normalization,
                normalization_protocol_sha256=object_digest(normalization), total=total)


def refresh_report(report):
    """Reaggregate existing AISBench per-sample outputs without scoring again."""
    if not report.get('input_signature') or not report.get('sample_ids') or not isinstance(report.get('details'), dict):
        raise ValueError('An AISBench report with input_signature, sample_ids and per-sample details is required')
    categories = {}
    for value in report['details'].values():
        for row in value.get('records', []):
            idx, cat = row['idx'], row.get('category', 'unknown')
            if idx in categories and categories[idx] != cat:
                raise ValueError(f'Inconsistent sample category: {idx}')
            categories[idx] = cat
    manifest = dict(input_signature=report['input_signature'],
                    evaluation_scope=report.get('evaluation_scope', 'legacy'),
                    records=[dict(idx=idx, category=categories.get(idx, 'unknown')) for idx in report['sample_ids']])
    result = build_report(manifest, report['details'], report.get('provenance', {}),
                          report.get('requested_dimensions', list(report['details'])))
    if report.get('postprocessing'):
        result['postprocessing'] = report['postprocessing']
    return result


def display_report(report):
    if report.get('input_signature') and report.get('sample_ids'):
        return refresh_report(report)
    # Metric-only legacy artifacts can display native means, but cannot certify a total.
    metrics = {}
    expected = max((m.get('expected', 0) for m in report.get('metrics', {}).values()), default=0)
    for dimension in DIMENSIONS:
        data = dict(report.get('metrics', {}).get(dimension, {}))
        data.setdefault('mean', None)
        for field in ('scored', 'missing', 'failed', 'not_evaluated'):
            data.setdefault(field, 0)
        data.setdefault('expected', expected)
        data.setdefault('scenes', {})
        data['omitted'] = data['missing'] if dimension == 'second_lsa' else 0
        data['coverage_percent'] = 100 * data['scored'] / data['expected'] if data['expected'] else 0.0
        data['status'] = 'incomplete'
        data.update(normalize_score(dimension, data['mean']))
        data.setdefault('out_of_bounds_count', 0)
        metrics[dimension] = data
    result = dict(report, metrics=metrics, directions={
        d: 'lower' if d == 'second_desync' else 'higher' for d in DIMENSIONS})
    result['total'] = aggregate(metrics)
    result['total'].update(score=None, status='unavailable', label='总分不可用',
                           note='Metric-only legacy report lacks auditable per-sample records')
    return result


def format_summary(report):
    rows = [f'VABench: scope={report.get("evaluation_scope", "legacy")}, '
            f'samples={len(report.get("sample_ids", []))}', '',
            '| 指标 | 原始均分 | 归一化分(0–100) | 有效/输入 | LSA省略 | 缺失(含省略) | 失败 | 覆盖率 | 状态 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for dimension, values in report['metrics'].items():
        raw = '—' if values['mean'] is None else f'{values["mean"]:.5f}'
        score = values.get('normalized_score')
        normalized = '—' if score is None else f'{score:.2f}'
        rows.append(f'| {dimension} | {raw} | {normalized} | {values["scored"]}/{values["expected"]} '
                    f'| {values.get("omitted", 0)} | {values["missing"]} | {values["failed"]} '
                    f'| {values["coverage_percent"]:.1f}% | {values["status"]} |')
    total = report['total']
    value = '—' if total['score'] is None else f'{total["score"]:.2f}/100'
    rows += ['', f'{total["label"]}: {value} '
             f'(有效指标 {total["valid_dimensions"]}/{total["expected_dimensions"]})',
             '总分采用固定归一化后的15项等权平均；这是AISBench展示协议，只代表当前选定样本。']
    if report.get('scope_note'):
        rows += [report['scope_note']]
    return '\n'.join(rows) + '\n'


def export_csv(report, output):
    output = Path(output)
    columns = ['mean', 'normalized_score', 'expected', 'scored', 'omitted', 'missing', 'failed',
               'not_evaluated', 'coverage_percent', 'out_of_bounds_count', 'mean_out_of_bounds', 'status']
    for filename, groups in [('metrics.csv', False), ('scenes.csv', True)]:
        with (output / filename).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['dimension'] + (['category'] if groups else ['direction']) + columns)
            for dim, values in report['metrics'].items():
                entries = values['scenes'].items() if groups else [(report['directions'][dim], values)]
                for category, data in entries:
                    writer.writerow([dim, category] + [data.get(k) for k in columns])
    with (output / 'samples.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dimension', 'idx', 'category', 'status', 'score', 'original_score', 'requested_device'])
        for dim, values in report['details'].items():
            for row in values['records']:
                writer.writerow([dim] + [row.get(k) for k in ('idx', 'category', 'status', 'score', 'original_score')]
                                + [values.get('requested_device', 'unobserved')])
    with (output / 'total.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['score', 'status', 'label', 'valid_dimensions', 'expected_dimensions'])
        writer.writerow([report['total'][k] for k in (
            'score', 'status', 'label', 'valid_dimensions', 'expected_dimensions')])
    (output / 'summary.md').write_text(format_summary(report), encoding='utf-8')


def _media_key(value):
    if not isinstance(value, (str, os.PathLike)):
        return None
    return os.path.normpath(os.fspath(value))


def _prepared_prompt_catalog(prepared):
    """Prompt text per category, keyed by the exact media paths prepared for it."""
    catalog = {}
    if prepared is None:
        return catalog
    root = Path(prepared) / 'json'
    if not root.is_dir():
        return catalog
    for path in sorted(root.glob('*.json')):
        try:
            entries = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(entries, list):
            continue
        entries = [entry for entry in entries if isinstance(entry, dict)]
        prompts, media = [], {}
        for entry in entries:
            prompt = entry.get('prompt_en')
            if not isinstance(prompt, str) or not prompt:
                continue
            if prompt not in prompts:
                prompts.append(prompt)
            for field in ('video_list', 'audio_list'):
                values = entry.get(field)
                if not isinstance(values, list):
                    continue
                for value in values:
                    key = _media_key(value)
                    if key is not None:
                        media[key] = prompt
        catalog[path.stem] = dict(prompts=prompts, media=media, entries=entries)
    return catalog


def _record_prompt(record, raw, media):
    for field in ('video_path', 'audio_path'):
        prompt = media.get(_media_key(raw.get(field)))
        if prompt is not None:
            return prompt
    for field in ('prompt_name', 'prompt_en', 'prompt'):
        prompt = raw.get(field)
        if isinstance(prompt, str) and prompt:
            return prompt
    return str(record.get('idx', ''))


def export_official_results(report, output, prepared=None, timestamp=None):
    """Back up raw scores in the official per-category JSON/CSV layout.

    ``video_results`` keeps every scorer record, including rows AISBench marks
    as parse/backend errors, so it can be diffed against official evaluate.py.
    The CSV repeats the official prompt-level aggregation and writes -1 for a
    prompt without a numeric score.
    """
    output = Path(output)
    stamp = timestamp or datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    catalog = _prepared_prompt_catalog(prepared)
    details = report.get('details', {})
    dimensions = list(details)
    categories = sorted({row.get('category', 'unknown')
                         for value in details.values()
                         for row in value.get('records', [])})
    for category in categories:
        folder = output / 'official_eval_results' / category
        name = f'{category}_{stamp}'
        folder.mkdir(parents=True, exist_ok=True)
        category_rows, results = {}, {}
        for dimension in dimensions:
            rows = [row for row in details[dimension].get('records', [])
                    if row.get('category', 'unknown') == category]
            category_rows[dimension] = rows
            results[dimension] = dict(video_results=[
                row['raw'] for row in rows if isinstance(row.get('raw'), dict)])
        write_json(folder / f'{name}_eval_results.json', results)
        prompt_info = catalog.get(category, {})
        prompts = list(prompt_info.get('prompts', []))
        media = prompt_info.get('media', {})
        scores = {}
        for dimension, rows in category_rows.items():
            buckets = scores.setdefault(dimension, {})
            for row in rows:
                raw = row.get('raw')
                if not isinstance(raw, dict):
                    continue
                score = raw.get('video_results')
                if not isinstance(score, (int, float)) or isinstance(score, bool):
                    continue
                prompt = _record_prompt(row, raw, media)
                if prompt not in prompts:
                    prompts.append(prompt)
                buckets.setdefault(prompt, []).append(float(score))
        with (folder / f'{name}_eval_results.csv').open(
                'w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['prompt_name'] + dimensions)
            for prompt in prompts:
                values = []
                for dimension in dimensions:
                    items = scores.get(dimension, {}).get(prompt, [])
                    values.append(sum(items) / len(items) if items else -1)
                writer.writerow([prompt] + values)
        if prompt_info.get('entries'):
            write_json(folder / f'{name}_full_info.json', prompt_info['entries'])
    return dict(directory=output / 'official_eval_results', timestamp=stamp,
                categories=categories)


def compare(reference, candidate, dimensions=None, relative=0.01,
            near_zero=1e-6, absolute=1e-6):
    """Same media AND protocol, source, weights, preprocessing, status sets.

    Different devices are allowed: this is also the GPU-vs-NPU audit tool.
    Each dataset mean is sample-weighted, never an average of scene means.
    """
    if not all(finite_number(v) for v in (relative, near_zero, absolute)) or relative <= 0 or near_zero < 0 or absolute < 0:
        raise ValueError('Invalid tolerance')
    dimensions = list(DIMENSIONS if dimensions is None else dimensions)
    if not dimensions or len(set(dimensions)) != len(dimensions):
        raise ValueError('Nonempty, unique dimensions required')
    problems = []
    for field in ('input_signature', 'sample_ids'):
        if not reference.get(field) or reference.get(field) != candidate.get(field):
            problems.append(f'{field} mismatch or missing')
    for key in ('protocol_sha256', 'source_signature', 'integration_signature',
                'weights_signature', 'scoring_signature'):
        left = reference.get('provenance', {}).get(key)
        right = candidate.get('provenance', {}).get(key)
        if not left or left != right:
            problems.append(f'{key} mismatch or missing')
    reports = {}
    for dim in dimensions:
        left = reference.get('details', {}).get(dim)
        right = candidate.get('details', {}).get(dim)
        issues = []
        a = b = None
        if not left or not right:
            issues.append('dimension missing')
        else:
            lrows, rrows = left['records'], right['records']
            ls = {r['idx']: r['status'] for r in lrows}
            rs = {r['idx']: r['status'] for r in rrows}
            if len(ls) != len(lrows) or len(rs) != len(rrows):
                issues.append('duplicate sample result')
            if ls != rs or sorted(ls) != reference.get('sample_ids', []):
                issues.append('sample/status set mismatch')
            if any(s not in ('ok', 'missing') for s in list(ls.values()) + list(rs.values())):
                issues.append('backend failure/invalid score')
            if dim != 'second_lsa' and ('missing' in ls.values() or 'missing' in rs.values()):
                issues.append('unexpected missing score')
            if not any(s == 'ok' for s in ls.values()):
                issues.append('no valid scores')
            if any(not finite_number(r.get('score')) for r in lrows + rrows if r['status'] == 'ok'):
                issues.append('nonfinite score')
            if not issues:
                a = mean(r['score'] for r in lrows if r['status'] == 'ok')
                b = mean(r['score'] for r in rrows if r['status'] == 'ok')
        delta = abs(a - b) if a is not None else None
        error = delta / abs(a) if a is not None and abs(a) > near_zero else None
        passed = (delta is not None and not issues and
                  (delta <= absolute if error is None else error < relative))
        reports[dim] = dict(passed=passed, reference_mean=a, candidate_mean=b,
                            absolute_error=delta, relative_error=error,
                            rule='absolute' if error is None else 'relative', issues=issues)
    return dict(passed=not problems and all(r['passed'] for r in reports.values()),
                protocol_issues=problems, dimensions=reports,
                thresholds=dict(relative_strictly_less_than=relative,
                                near_zero_at_most=near_zero, absolute_at_most=absolute))

								
								
work.py

"""Isolated official model process, with an optional Ascend migration shim."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import importlib.metadata
import functools
import json
import logging
import os
from pathlib import Path
import re
import sys
import traceback

from .data import read_json, write_json
from .resources import submodules
from .results import normalize_records


def initialize_device(device):
    import torch
    if device.startswith('npu'):
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            raise RuntimeError('Requested NPU is unavailable')
        # Only this child process sees the CUDA migration shim.
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
        torch.npu.set_device(device)
    elif device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable')
        torch.cuda.set_device(device)
    elif device != 'cpu':
        raise ValueError(f'Unsupported device {device}')
    return torch.device(device)


def _getattr_or_none(obj, name):
    """Read an attribute without triggering exotic __getattr__ side effects."""
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _observe_devices(module):
    """Inspect retained torch modules and ONNX sessions, without guessing."""
    devices, providers, seen = set(), set(), set()

    def walk(obj, depth=0):
        if id(obj) in seen or depth > 3:
            return
        seen.add(id(obj))
        # ctypes.LibraryLoader.__getattr__ tries to dlopen the attribute name;
        # a plain hasattr(obj, 'parameters') can therefore raise OSError.
        parameters = _getattr_or_none(obj, 'parameters')
        if callable(parameters):
            try:
                devices.update(str(p.device) for p in parameters())
            except Exception:
                pass
        get_providers = _getattr_or_none(obj, 'get_providers')
        if callable(get_providers):
            try:
                providers.update(get_providers())
            except Exception:
                pass
        if isinstance(obj, (list, tuple)):
            for value in obj:
                walk(value, depth + 1)
        elif isinstance(obj, dict):
            for value in obj.values():
                walk(value, depth + 1)
        elif not isinstance(obj, type):
            try:
                attrs = vars(obj)
            except TypeError:
                attrs = None
            if attrs is not None:
                for name, value in attrs.items():
                    if not name.startswith('__'):
                        walk(value, depth + 1)
    for name, obj in vars(module).items():
        if 'MODEL' in name.upper() or 'CACHE' in name.upper():
            walk(obj)
    return dict(observed_torch_devices=sorted(devices), observed_onnx_providers=sorted(providers))


def _rebind_stream_handlers(old_stream, new_stream):
    """Move logging handlers off a redirect target that is about to close.

    A library that calls logging.basicConfig() while stderr is redirected
    leaves a handler pointing at the per-dimension backend.log. Logging to it
    after the file closes prints a traceback, which used to be misread as a
    metric failure.
    """
    loggers = [logging.getLogger()]
    loggers.extend(obj for obj in logging.Logger.manager.loggerDict.values()
                   if isinstance(obj, logging.Logger))
    for logger in loggers:
        for handler in list(logger.handlers):
            if _getattr_or_none(handler, 'stream') is old_stream:
                handler.acquire()
                try:
                    handler.stream = new_stream
                finally:
                    handler.release()


def instrument_models(module, observations, device):
    """Observe model factories before official cleanup drops their objects."""
    from types import SimpleNamespace
    if module.__name__.endswith('first_audiobox'):
        original_predictor = module.initialize_predictor
        def predictor_on_device(*args, **kwargs):
            predictor = original_predictor(*args, **kwargs)
            # The upstream factory chooses CUDA automatically and ignores
            # compute_*'s device. Explicit placement is essential for CPU.
            import torch
            predictor.device = torch.device(device)
            predictor.model.to(predictor.device)
            return predictor
        if not getattr(original_predictor, '_vabench_observer', False):
            predictor_on_device._vabench_observer = True
            module.initialize_predictor = predictor_on_device
    if module.__name__.endswith('second_lsa') and not device.startswith('cuda'):
        # InsightFace is an ONNX subcomponent; explicitly use CPU on A3.
        # Preserve the official detector models, det_size and allowed modules.
        face = importlib.import_module('third_party.syncnet_eval.latentsync.utils.face_detector')
        def initialize_face(self, device='cpu'):
            self.app = face.FaceAnalysis(
                allowed_modules=['detection', 'landmark_2d_106'],
                root=face._INSIGHTFACE_ROOT, providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=-1, det_size=(face.INSIGHTFACE_DETECT_SIZE,
                                                face.INSIGHTFACE_DETECT_SIZE))
            providers = set()
            for model in getattr(self.app, 'models', {}).values():
                session = getattr(model, 'session', None)
                if session is not None:
                    providers.update(session.get_providers())
            observations.append(dict(observed_torch_devices=[],
                                     observed_onnx_providers=sorted(providers)))
        face.FaceDetector.__init__ = initialize_face
        observations.append(dict(component='InsightFace', configured_device='cpu',
                                 reason='Explicit ONNX CPU subcomponent on non-CUDA runtime'))
    for name, func in list(vars(module).items()):
        if callable(func) and name.startswith(('_get_', '_load_')) and not getattr(func, '_vabench_observer', False):
            @functools.wraps(func)
            def wrapped(*args, _func=func, **kwargs):
                result = _func(*args, **kwargs)
                if module.__name__.endswith('first_dnsmos') and device == 'cpu':
                    # ONNX execution providers are independent of torch.device.
                    for value in vars(result).values() if hasattr(result, '__dict__') else []:
                        if hasattr(value, 'set_providers'):
                            value.set_providers(['CPUExecutionProvider'])
                observations.append(_observe_devices(SimpleNamespace(_MODEL=result)))
                return result
            wrapped._vabench_observer = True
            setattr(module, name, wrapped)


def has_backend_error(log_text):
    return bool(re.search(
        r'Traceback \(most recent|OOM_ERROR|out of memory|'
        r'^\s*(?:Error\b|Failed\b|\[.*?\]\s*Error\b)|'
        r'处理音频文件失败|视频评估失败', log_text, re.I | re.M))


def failed_sample_ids(log_text):
    """Samples named by official per-video error messages.

    VABench prints ``Error processing <path>`` (CLAP, Desync) or
    ``Error processing sample <path>`` (ImageBind) and then keeps looping.
    The remaining results in that scene stay valid.
    """
    return {Path(match.group(1).strip()).stem for match in re.finditer(
        r'Error processing (?:sample )?(.+?):(?:\s|$)', log_text)}


def execute(request):
    root = Path(request['source_root']).resolve()
    os.environ['VABENCH_CACHE_DIR'] = request['cache_dir']
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(root))
    device = initialize_device(request['device'])
    import torch
    omni = importlib.import_module('vabench.omni_runtime')
    # Capture model residency when loaded, before official release_omni()
    # discards its model. Do not change processor, decoding, or model weights.
    observed_omni = set()
    original_get = omni.get_omni_model_and_processor

    def observe_omni(*args, **kwargs):
        model, processor = original_get(*args, **kwargs)
        observed_omni.update(str(p.device) for p in model.parameters())
        return model, processor

    omni.get_omni_model_and_processor = observe_omni
    grouped = request['grouped']
    expected = request['records']
    details = {}
    for dimension in request['dimensions']:
        records, errors, residency, factory_observations = [], [], [], []
        for category, metadata in grouped.items():
            scene_expected = [r for r in expected if r['category'] == category]
            sandbox = Path(request['work_dir']) / dimension / category
            sandbox.mkdir(parents=True, exist_ok=True)
            full_info = sandbox / 'full_info.json'
            write_json(full_info, metadata)
            old_cwd = Path.cwd()
            log_file = sandbox / 'backend.log'
            raw, observed, caught = [], {}, None
            log = None
            try:
                os.chdir(sandbox)
                with log_file.open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
                    module = importlib.import_module(f'vabench.{dimension}')
                    instrument_models(module, factory_observations, request['device'])
                    params = dict(request.get('params', {}))
                    params.update(request.get('dim_params', {}).get(dimension, {}))
                    if dimension.startswith(('third_', 'fourth_')):
                        params['qwen_omni_ckpt'] = str(Path(request['cache_dir']) / 'Qwen2.5_omni_7B')
                        params['local_files_only'] = True
                        if request['device'].startswith('npu'):
                            params['device_map'] = {'': request['device']}
                            params.setdefault('attn_implementation', 'eager')
                    if request['mode'] == 'official':
                        raw = _official_scene(request, dimension, metadata, sandbox, device, params)
                    else:
                        result = getattr(module, f'compute_{dimension}')(
                            str(full_info), device, submodules(dimension, request['cache_dir']), **params)
                        raw = result[1]
                    observed = _observe_devices(module)
            except Exception:
                caught = traceback.format_exc()
                errors.append(dict(category=category, error=caught))
            finally:
                os.chdir(old_cwd)
                if log is not None:
                    # The with-block restored sys.stderr; detach handlers that
                    # were created while it pointed at this closed log file.
                    _rebind_stream_handlers(log, sys.stderr)
            text = log_file.read_text(encoding='utf-8') if log_file.exists() else ''
            bad_log = has_backend_error(text)
            named_failures = failed_sample_ids(text) & {r['idx'] for r in scene_expected}
            if bad_log and caught is None:
                errors.append(dict(category=category,
                                   error='Backend reported error; inspect backend.log',
                                   samples=sorted(named_failures)))
            try:
                all_failed = bool(caught or (bad_log and not named_failures))
                records.extend(normalize_records(
                    dimension, raw, scene_expected, backend_errors=all_failed,
                    failed_ids=None if all_failed else named_failures))
            except Exception:
                errors.append(dict(category=category, error=traceback.format_exc()))
                records.extend(normalize_records(dimension, [], scene_expected, backend_errors=True))
            residency.append(dict(category=category, **observed))
        details[dimension] = dict(records=records, errors=errors,
                                  requested_device=request['device'], device_observations=residency,
                                  model_factory_observations=factory_observations)
        write_json(Path(request['work_dir']) / 'partial.json', details)
    for dim in details:
        if dim.startswith(('third_', 'fourth_')):
            details[dim]['observed_omni_devices'] = sorted(observed_omni)
    omni.release_omni()
    versions = {}
    for package in ('torch', 'torch-npu', 'transformers', 'onnxruntime', 'onnxruntime-gpu', 'qwen-omni-utils'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(details=details, environment=dict(python=sys.version, packages=versions))


def _official_scene(request, dimension, metadata, sandbox, device, params):
    """Run the official class, fixing only its incorrect mapping lookup path.

    Official filenames are recreated as symlinks in an isolated cwd. Linux is
    the supported model runtime. Media content and metadata stay identical.
    """
    import csv
    from vabench import VABench
    from vabench import utils
    # Avoid official init_submodules' network downloads. The adapter has
    # already required the same explicit, local cache paths.
    import vabench
    vabench.init_submodules = lambda dims, local=False: {
        d: submodules(d, request['cache_dir']) for d in dims}
    mapping_file = Path(request['source_root']) / 'mapping/final_idx_to_prompt.csv'
    with mapping_file.open(encoding='utf-8', newline='') as stream:
        mapping = {r['idx']: json.loads('"' + r['prompt'] + '"')
                   for r in csv.DictReader(stream, delimiter='\t')}

    class OfficialWithMappingPath(VABench):
        def load_prompt_mapping(self):
            return mapping

    for media in ('video', 'audio'):
        folder = sandbox / media
        folder.mkdir(exist_ok=True)
        for item in metadata:
            for source in item[f'{media}_list']:
                target = folder / Path(source).name
                if not target.exists():
                    try:
                        target.symlink_to(source)
                    except OSError:
                        # Windows without symlink privileges, same filesystem.
                        target.hardlink_to(source)
    model = OfficialWithMappingPath(device, str(sandbox / 'full_info.json'), str(sandbox))
    model.evaluate(str(sandbox / 'video'), str(sandbox / 'audio'),
                   'official', dimension_list=[dimension], local=True, **params)
    results = utils.load_json(str(sandbox / 'official_eval_results.json'))
    return results[dimension]['video_results']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request')
    args = parser.parse_args()
    request = read_json(args.request)
    try:
        result = execute(request)
        write_json(request['response'], result)
    except Exception:
        write_json(request['response'], dict(error=traceback.format_exc()))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
