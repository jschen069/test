"""AISBench registry, task config, summary and packaging contracts."""
from pathlib import Path
import io

from mmengine.config import Config, ConfigDict

from ais_bench.benchmark.tasks.vabench_eval import VABenchEvalTask
from ais_bench.benchmark.summarizers.vabench import VABenchSummarizer
from ais_bench.benchmark.vabench.data import write_json


def test_config_and_partitioning(tmp_path):
    from ais_bench.benchmark.partitioners import NaivePartitioner
    cfg = Config.fromfile('ais_bench/configs/vabench_examples/eval_vabench.py')
    cfg.work_dir = str(tmp_path)
    cfg.cli_args = {'mode': 'eval', 'debug': False}
    task_cfg = NaivePartitioner(keep_keys=['eval.runner.task.num_gpus'])(cfg)[0]
    task = VABenchEvalTask(task_cfg)
    assert task.num_gpus == 1
    assert len(task.dataset_cfgs) == 1
    command = task.get_command('config with spaces.py', '{task_cmd}')
    assert 'ais_bench.benchmark.tasks.vabench_eval' in command
    assert 'config with spaces.py' in command


def test_smoke_config_selects_only_clap():
    cfg = Config.fromfile('ais_bench/configs/vabench_examples/eval_vabench_smoke.py')
    assert cfg.datasets[0].eval_cfg.dimensions == ['second_clap']
    assert 'smoke' in cfg.models[0].abbr
    assert 'smoke' in cfg.datasets[0].abbr


def test_task_preserves_incomplete_report(tmp_path, monkeypatch):
    import ais_bench.benchmark.tasks.vabench_eval as task_module
    report = dict(metrics={'second_clap': dict(scored=0, failed=1, missing=0)}, details={})
    monkeypatch.setattr(task_module, 'run', lambda *a, **kw: report)
    prepared = tmp_path / 'prepared'
    write_json(prepared / 'manifest.json', dict(records=[]))
    cfg = ConfigDict(models=[dict(abbr='test')], datasets=[[
        dict(abbr='vabench', path=str(prepared),
             eval_cfg=dict(source_root='source', cache_dir='cache'))]],
        work_dir=str(tmp_path), cli_args={})
    import pytest
    with pytest.raises(RuntimeError, match='incomplete'):
        VABenchEvalTask(cfg).run()
    assert (tmp_path / 'results/test/vabench.json').exists()


def test_native_summary_with_normalization_and_unavailable_total(tmp_path):
    cfg = ConfigDict(models=[dict(abbr='test')],
                     datasets=[dict(abbr='vabench', path='prepared')], work_dir=str(tmp_path))
    write_json(tmp_path / 'results/test/vabench.json', dict(
        metrics={'second_desync': dict(mean=0.2, expected=10, scored=10, missing=0,
                                       failed=0, scenes={'scene': {'mean': 0.2}})},
        directions={'second_desync': 'lower'}, details={}))
    summarizer = VABenchSummarizer(cfg)
    raw, parsed, metrics, modes = summarizer._pick_up_results()
    assert parsed['test']['vabench/second_desync']['mean'] == 0.2
    assert raw['test']['vabench/second_desync']['direction'] == 'lower'
    assert parsed['test']['vabench/second_desync']['normalized_score'] == 90
    assert parsed['test']['vabench/total']['normalized_score'] == '—'
    assert len(parsed['test']) == 16
    table = summarizer._format_table(parsed, metrics, modes,
                                     required_dataset_abbrs=summarizer.dataset_abbrs)
    assert table
    row = next(r for r in table if len(r) > 1 and r[1] == 'vabench/second_desync')
    assert row[2] == 'lower'
    # A failed dimension has an explicit N/A mean, never a fabricated zero.
    file = tmp_path / 'results/test/vabench.json'
    from ais_bench.benchmark.vabench.data import read_json
    report = read_json(file)
    report['metrics']['second_desync'].update(mean=None, scored=0, failed=10)
    write_json(file, report)
    raw, parsed, metrics, modes = summarizer._pick_up_results()
    table = summarizer._format_table(parsed, metrics, modes)
    row = next(r for r in table if len(r) > 1 and r[1] == 'vabench/second_desync')
    assert row[3] == '—' and row[-1] == 'incomplete'



"""CPU-only contract tests: real FFmpeg, real subprocess and data audits."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ais_bench.benchmark.vabench import DIMENSIONS, MLLM_DIMENSIONS
from ais_bench.benchmark.vabench.data import (
    PROTOCOL_PATH, catalog, catalog_videos, digest, media_tool, prepare, read_json,
    verify_prepared, write_json)
from ais_bench.benchmark.vabench.engine import run, successful
from ais_bench.benchmark.vabench.results import (
    build_report, compare, normalize_records, summarize_dimension)
from ais_bench.benchmark.vabench.worker import failed_sample_ids, has_backend_error


SOURCE = Path(__file__).resolve().parents[4] / 'VABench'


def test_pinned_catalog():
    if not SOURCE.exists():
        pytest.skip('Local official checkout is not available')
    standard, mini = catalog(SOURCE), catalog(SOURCE, 'mini')
    assert len(standard) == 778 and len(mini) == 10
    assert len({e['idx'] for e in standard}) == 778
    assert {e['idx'] for e in mini} <= {e['idx'] for e in standard}
    assert len({e['category'] for e in mini}) == 10
    assert {'easy_human_language', 'hard_human_language', 'hard_Exact_Time_Synchronization'} <= {
        e['category'] for e in mini}


def test_pinned_data_rejects_drift(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    with pytest.raises(ValueError, match='differs from pinned'):
        catalog(root)


def test_explicit_single_sample_smoke_catalog():
    if not SOURCE.exists():
        pytest.skip('Local official checkout is not available')
    entries = catalog(SOURCE, 'smoke', sample_id='kitten curled up meowing')
    assert len(entries) == 1 and entries[0]['idx'] == 'kitten curled up meowing'
    assert len(catalog(SOURCE, 'mini')) == 10
    assert len(catalog(SOURCE, 'standard')) == 778
    with pytest.raises(ValueError, match='requires one sample_id'):
        catalog(SOURCE, 'smoke')
    with pytest.raises(ValueError, match='requires one sample_id'):
        catalog(SOURCE, 'smoke', sample_id='not-an-official-prompt')
    with pytest.raises(ValueError, match='only supported'):
        catalog(SOURCE, 'mini', sample_id='kitten curled up meowing')


def test_catalog_videos_selects_pinned_samples(tmp_path):
    if not SOURCE.exists():
        pytest.skip('Local official checkout is not available')
    idx = catalog(SOURCE, 'mini')[0]['idx']
    (tmp_path / f'{idx}.mp4').write_bytes(b'')
    selected = catalog_videos(SOURCE, tmp_path)
    assert [entry['idx'] for entry in selected] == [idx]
    (tmp_path / 'not an official sample.mp4').write_bytes(b'')
    with pytest.raises(ValueError, match='outside the pinned 778'):
        catalog_videos(SOURCE, tmp_path)


def test_catalog_videos_rejects_ambiguous_stems(tmp_path):
    if not SOURCE.exists():
        pytest.skip('Local official checkout is not available')
    idx = catalog(SOURCE, 'mini')[0]['idx']
    for name in ('a', 'b'):
        folder = tmp_path / name
        folder.mkdir()
        (folder / f'{idx}.mp4').write_bytes(b'')
    with pytest.raises(ValueError, match='Ambiguous MP4 stems'):
        catalog_videos(SOURCE, tmp_path)


@pytest.fixture
def entry():
    return dict(idx='中文 sample with spaces', category='scene', metadata=dict(
        prompt_en='A test tone with a visual scene.', prompt_audio='A tone.',
        prompt_vision='A color image.', dimension=list(DIMENSIONS),
        auxiliary_info_audio=['Is a tone audible?'], auxiliary_info_vision=['Is color visible?']))


@pytest.fixture
def video(tmp_path, entry):
    if not shutil.which(media_tool('ffmpeg')) or not shutil.which(media_tool('ffprobe')):
        pytest.skip('FFmpeg and ffprobe are required for media tests')
    folder = tmp_path / 'videos'
    folder.mkdir()
    file = folder / (entry['idx'] + '.mp4')
    subprocess.run([
        media_tool('ffmpeg'), '-v', 'error', '-y', '-f', 'lavfi', '-i',
        'color=c=blue:s=64x64:r=24:d=0.3', '-f', 'lavfi', '-i',
        'sine=frequency=440:sample_rate=48000:duration=0.3',
        '-c:v', 'mpeg4', '-c:a', 'aac', '-shortest', str(file)], check=True)
    return file


def test_real_media_preparation(tmp_path, entry, video):
    out = tmp_path / 'prepared'
    manifest = prepare([entry], video.parent, out)
    assert manifest['subset_size'] == 1
    assert manifest['records'][0]['short_video'] is True
    verified, grouped = verify_prepared(out)
    assert verified['input_signature'] == manifest['input_signature']
    assert grouped['scene'][0]['prompt_en'] == entry['metadata']['prompt_en']
    wave = Path(manifest['records'][0]['audio_path'])
    import wave as wave_module
    with wave_module.open(str(wave)) as audio:
        assert audio.getframerate() == 48000
        assert audio.getsampwidth() == 2
    with pytest.raises(FileExistsError):
        prepare([entry], video.parent, out)
    wave.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Input changed'):
        verify_prepared(out)


def test_smoke_cli_labels_single_sample(tmp_path, entry, video, monkeypatch):
    import ais_bench.benchmark.vabench.cli as cli
    monkeypatch.setattr(cli, 'catalog', lambda *a, **kw: [entry])
    out = tmp_path / 'smoke'
    assert cli.main(['prepare', '--source-root', 'fixture', '--subset', 'smoke',
                     '--sample-id', entry['idx'], '--videos', str(video.parent),
                     '--output', str(out)]) == 0
    manifest, grouped = verify_prepared(out)
    assert manifest['evaluation_scope'] == 'smoke'
    assert manifest['subset_size'] == 1
    report = build_report(manifest, {}, {})
    assert report['evaluation_scope'] == 'smoke'
    assert 'not Mini10/778' in report['scope_note']
    with pytest.raises(ValueError, match='exactly one'):
        prepare([entry, entry], video.parent, tmp_path / 'invalid', subset='smoke')


def test_missing_ambiguous_and_silent_input(tmp_path, entry, video):
    with pytest.raises(ValueError, match='Missing MP4'):
        prepare([dict(entry, idx='missing')], video.parent, tmp_path / 'out1')
    extra = video.parent / 'duplicate'
    extra.mkdir()
    shutil.copyfile(video, extra / video.name)
    with pytest.raises(ValueError, match='ambiguous MP4'):
        prepare([entry], video.parent, tmp_path / 'out2')
    (extra / video.name).unlink()
    silent = video.parent / 'silent.mp4'
    subprocess.run([media_tool('ffmpeg'), '-v', 'error', '-y', '-i', str(video),
                    '-an', '-c:v', 'copy', str(silent)], check=True)
    with pytest.raises(ValueError, match='must contain video and audio'):
        prepare([dict(entry, idx='silent')], video.parent, tmp_path / 'out3')


def _report(score=0.0, status='ok', dim='second_clap'):
    records = [dict(idx='x', category='scene', status=status, score=score, raw={})]
    return build_report(dict(input_signature='same', records=[{'idx': 'x'}]),
                        {dim: {'records': records}},
                        {k: 'same' for k in ('protocol_sha256', 'source_signature', 'integration_signature',
                                             'weights_signature', 'scoring_signature')})


def test_comparison_zero_negative_and_threshold():
    reference = _report(0)
    assert compare(reference, _report(0), ['second_clap'])['passed']
    assert not compare(reference, _report(0.01), ['second_clap'])['passed']
    assert compare(_report(-0.5), _report(-0.502), ['second_clap'])['passed']
    # Boundary uses strict < 1%, rather than <= 1%.
    assert not compare(_report(100), _report(101), ['second_clap'])['passed']
    assert not compare(_report(), _report())['passed']  # fifteen dims required by default
    with pytest.raises(ValueError, match='Invalid tolerance'):
        compare(reference, reference, relative=float('inf'))


@pytest.mark.parametrize('mutation', ['missing', 'failed', 'weights', 'media', 'duplicate', 'nan'])
def test_alignment_cannot_pass_invalid_runs(mutation):
    left, right = _report(0.5), _report(0.5)
    if mutation == 'missing':
        right['details']['second_clap']['records'][0]['status'] = 'missing'
    elif mutation == 'failed':
        for report in (left, right):
            report['details']['second_clap']['records'][0]['status'] = 'backend_error'
    elif mutation == 'weights':
        right['provenance']['weights_signature'] = 'different'
    elif mutation == 'media':
        right['input_signature'] = 'different'
    elif mutation == 'duplicate':
        right['details']['second_clap']['records'] *= 2
    elif mutation == 'nan':
        right['details']['second_clap']['records'][0]['score'] = float('nan')
    assert not compare(left, right, ['second_clap'])['passed']


def test_statuses_and_weighted_mean():
    expected = [dict(idx='x', category='a'), dict(idx='y', category='b')]
    rows = normalize_records('second_lsa', [{'video_path': 'x.mp4', 'video_results': 0}], expected)
    assert [r['status'] for r in rows] == ['ok', 'missing']
    summary = summarize_dimension(rows)
    assert summary['mean'] == 0 and summary['missing'] == 1
    rows = [dict(idx=str(i), category=cat, status='ok', score=score)
            for i, (cat, score) in enumerate([('a', 0), ('a', 0), ('b', 3)])]
    assert summarize_dimension(rows)['mean'] == 1  # not (0+3)/2
    assert normalize_records('second_clap', [{'video_path': 'x.mp4', 'video_results': 0}],
                             expected, backend_errors=True)[0]['status'] == 'backend_error'
    assert has_backend_error('Error processing clip: failed')
    assert has_backend_error('Traceback (most recent call last):')
    assert not has_backend_error('Evaluation completed normally')
    assert not has_backend_error('人脸检测失败或评估失败: example.mp4')


def test_named_sample_failure_does_not_poison_scene():
    expected = [dict(idx='bad', category='c'), dict(idx='good', category='c')]
    rows = normalize_records('second_desync', [
        dict(video_path='/v/bad.mp4', video_results=1.0),
        dict(video_path='/v/good.mp4', video_results=0.25)],
        expected, failed_ids={'bad'})
    assert [r['status'] for r in rows] == ['backend_error', 'ok']
    assert rows[0]['score'] is None and rows[1]['score'] == 0.25
    assert failed_sample_ids(
        'Error processing /v/bad.mp4: tensor mismatch\nTraceback (most recent call last):\n') == {'bad'}
    # tqdm may write its progress bar and the error on the same line.
    assert failed_sample_ids(' 80%|####  | 4/5]Error processing /v/bad.mp4: boom\n') == {'bad'}
    assert failed_sample_ids('Error processing sample /v/bad.mp4: boom\n') == {'bad'}
    assert failed_sample_ids('Error in second_desync evaluation: boom\n') == set()


def test_case_progress_mapping():
    from ais_bench.benchmark.tasks.vabench_eval import _case_progress
    assert [_case_progress(7, done, 9) for done in range(10)] == [0, 1, 2, 2, 3, 4, 5, 5, 6, 7]
    assert _case_progress(0, 1, 2) == 0
    assert _case_progress(7, 0, 0) == 0


def test_task_progress_uses_sample_count(tmp_path, entry, video, monkeypatch):
    from ais_bench.benchmark.tasks import vabench_eval as task_module
    prepared = tmp_path / 'prepared'
    second = copy.deepcopy(entry)
    second['idx'] = 'second progress sample'
    shutil.copyfile(video, video.parent / (second['idx'] + '.mp4'))
    prepare([entry, second], video.parent, prepared, subset='custom')

    def fake_run(prepared, source, cache, destination, progress=None, **options):
        progress(1, 2, ['second_desync'])
        progress(2, 2, ['third_alignment'])
        return dict(requested_dimensions=['second_desync'], details={}, metrics={})

    monkeypatch.setattr(task_module, 'run', fake_run)
    monkeypatch.setattr(task_module, 'successful', lambda report: True)

    class Recorder:
        def __init__(self):
            self.states = []

        def update_task_state(self, state):
            self.states.append(dict(state))

    recorder = Recorder()
    cfg = dict(
        models=[dict(abbr='model', type='VABenchEvalPlaceholder', attr='local')],
        datasets=[[dict(abbr='vabench_t2av', path=str(prepared),
                        eval_cfg=dict(source_root='src', cache_dir='cache'))]],
        work_dir=str(tmp_path / 'work'), cli_args={})
    task_module.VABenchEvalTask(cfg).run(recorder)
    assert [(s['finish_count'], s['total_count']) for s in recorder.states] == [
        (0, 2), (1, 2), (2, 2), (2, 2)]
    assert recorder.states[0]['progress_description'] == ''
    assert recorder.states[-1]['other_kwargs'] == {'dimension': 'done'}


def test_explicit_cpu_onnx_provider_and_observation(monkeypatch):
    from types import ModuleType
    from ais_bench.benchmark.vabench.worker import instrument_models
    class Session:
        def __init__(self):
            self.providers = ['CUDAExecutionProvider']
        def set_providers(self, providers):
            self.providers = providers
        def get_providers(self):
            return self.providers
    class Scorer:
        def __init__(self):
            self.session = Session()
    module = ModuleType('vabench.first_dnsmos')
    module._load_dns_mos_model = lambda: Scorer()
    observations = []
    instrument_models(module, observations, 'cpu')
    scorer = module._load_dns_mos_model()
    assert scorer.session.providers == ['CPUExecutionProvider']
    assert observations[0]['observed_onnx_providers'] == ['CPUExecutionProvider']


def test_device_observer_tolerates_ctypes_library_loader():
    import ctypes
    from types import ModuleType, SimpleNamespace
    from ais_bench.benchmark.vabench.worker import _observe_devices
    module = ModuleType('vabench.probe')
    module.MODEL_CACHE = SimpleNamespace(loader=ctypes.cdll)
    assert _observe_devices(module) == {
        'observed_torch_devices': [], 'observed_onnx_providers': []}


def test_logging_handlers_are_rebound_after_redirect_close(tmp_path):
    import logging
    from contextlib import redirect_stderr
    from ais_bench.benchmark.vabench.worker import _rebind_stream_handlers
    logger = logging.getLogger('vabench-test-rebind')
    logger.handlers = []
    logger.propagate = False
    redirected = (tmp_path / 'backend.log').open('w')
    try:
        with redirect_stderr(redirected):
            handler = logging.StreamHandler()
            logger.addHandler(handler)
    finally:
        redirected.close()
    try:
        assert handler.stream.closed
        _rebind_stream_handlers(redirected, sys.stderr)
        assert handler.stream is sys.stderr
        logger.warning('this must not raise a logging error')
    finally:
        logger.removeHandler(handler)


def test_custom_split_validation(monkeypatch):
    import ais_bench.benchmark.vabench.engine as engine
    monkeypatch.setattr(engine, 'catalog', lambda root, subset: (
        [{'idx': 'a'}, {'idx': 'b'}] if subset == 'standard' else [{'idx': 'm'}]))
    engine._validate_prepared_split(
        dict(evaluation_scope='custom', records=[dict(idx='b')]), 'unused')
    with pytest.raises(ValueError, match='outside the pinned 778'):
        engine._validate_prepared_split(
            dict(evaluation_scope='custom', records=[dict(idx='z')]), 'unused')
    with pytest.raises(ValueError, match='exactly one'):
        engine._validate_prepared_split(
            dict(evaluation_scope='smoke', records=[dict(idx='a'), dict(idx='b')]), 'unused')
    with pytest.raises(ValueError, match='pinned 778/10'):
        engine._validate_prepared_split(
            dict(evaluation_scope='standard', records=[dict(idx='a')]), 'unused')


def test_custom_scope_is_labeled_non_official():
    from ais_bench.benchmark.vabench.results import build_report
    scores = dict(first_dnsmos=3, first_nisqa=3, first_audiobox=2.75,
                  second_viclip=0, second_clap=0, second_desync=1, second_lsa=5,
                  second_imagebind=0, third_alignment=3, third_audio_reality=3,
                  third_visual_reality=3, third_expressiveness=3, third_artistry=3,
                  fourth_qa_audio=0.5, fourth_qa_vision=0.5)
    details = {dim: dict(records=[dict(idx='x', category='a', status='ok', score=score,
                                       original_score=score, raw={})], errors=[])
               for dim, score in scores.items()}
    manifest = dict(input_signature='sig', evaluation_scope='custom',
                    records=[dict(idx='x', category='a')])
    report = build_report(manifest, details, {})
    assert report['evaluation_scope'] == 'custom'
    assert 'Custom subset of 1' in report['scope_note']
    assert 'not Mini10/778' in report['scope_note']
    assert report['total']['status'] == 'formal'
    assert report['total']['label'] == '自定义子集总分（非官方）'


def test_lsa_missing_sets_must_match_and_not_be_empty():
    left = _report(2.0, dim='second_lsa')
    left['details']['second_lsa']['records'].append(dict(idx='y', category='scene',
                                                       status='missing', score=None))
    left['sample_ids'] = ['x', 'y']
    assert compare(left, copy.deepcopy(left), ['second_lsa'])['passed']
    right = copy.deepcopy(left)
    right['details']['second_lsa']['records'][1]['status'] = 'ok'
    right['details']['second_lsa']['records'][1]['score'] = 2.0
    assert not compare(left, right, ['second_lsa'])['passed']


def test_real_worker_subprocess_and_mllm_grouping(tmp_path, entry, video, monkeypatch):
    """A fake scorer tests process/protocol plumbing, not model accuracy."""
    import ais_bench.benchmark.vabench.engine as engine
    prepared = tmp_path / 'prepared'
    prepare([entry], video.parent, prepared, subset='smoke')
    source = tmp_path / 'fake_source'
    package = source / 'vabench'
    package.mkdir(parents=True)
    (source / 'torch.py').write_text('def device(x): return x\n', encoding='utf-8')
    (package / '__init__.py').write_text('', encoding='utf-8')
    (package / 'omni_runtime.py').write_text(
        'def release_omni(): pass\n'
        'def get_omni_model_and_processor(*a, **kw): raise RuntimeError("unused")\n', encoding='utf-8')
    dims = ['third_alignment', 'fourth_qa_audio']
    for dim in dims:
        fixture_score = 1.0 if dim.startswith('fourth_') else 3.0
        (package / f'{dim}.py').write_text(
            'import json\n'
            f'def compute_{dim}(file, device, submodules, **kw):\n'
            '    items = json.load(open(file, encoding="utf-8"))\n'
            f'    return None, [{{"video_path": x["video_list"][0], "video_results": {fixture_score}}} for x in items]\n',
            encoding='utf-8')
    cache = tmp_path / 'cache'
    checkpoint = cache / 'Qwen2.5_omni_7B'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'dummy').write_text('test fixture, not a model')
    monkeypatch.setattr(engine, 'catalog', lambda *args: [entry])
    progress = []
    report = run(prepared, source, cache, tmp_path / 'run', dimensions=dims,
                 device='cpu', python=sys.executable,
                 progress=lambda done, total, names: progress.append((done, total, tuple(names))))
    assert successful(report)
    assert progress == [(1, 1, tuple(dims))]
    assert report['evaluation_scope'] == 'smoke'
    assert report['metrics']['third_alignment']['mean'] == 3
    assert len(list((tmp_path / 'run').glob('job_*'))) == 1
    assert len(list((tmp_path / 'run' / 'official_eval_results').glob('*/*_eval_results.json'))) == 1
    assert len(list((tmp_path / 'run' / 'official_eval_results').glob('*/*_eval_results.csv'))) == 1
    request = read_json(tmp_path / 'run/job_00/request.json')
    assert request['dimensions'] == dims
    assert compare(report, report, dims)['passed']

    # Exercise the real upstream VABench orchestration class, with the same
    # fake compute_* scorers. Only model inference is substituted.
    if SOURCE.exists():
        (package / '__init__.py').write_text(
            (SOURCE / 'vabench/__init__.py').read_text(encoding='utf-8'), encoding='utf-8')
        (source / 'torch.py').write_text(
            'def device(x): return x\n'
            'class cuda:\n'
            '    @staticmethod\n'
            '    def is_available(): return False\n', encoding='utf-8')
        (package / 'utils.py').write_text(
            'import json\n'
            'def init_submodules(*a, **kw): return {}\n'
            'def load_json(p):\n'
            '    with open(p, encoding="utf-8") as f: return json.load(f)\n'
            'def save_json(obj, p):\n'
            '    with open(p, "w", encoding="utf-8") as f: json.dump(obj, f)\n', encoding='utf-8')
        mapping_dir = source / 'mapping'
        mapping_dir.mkdir()
        (mapping_dir / 'final_idx_to_prompt.csv').write_text(
            'idx\tprompt\n' + entry['idx'] + '\t' + entry['metadata']['prompt_en'] + '\n',
            encoding='utf-8')
        # The source signature changed to install the real orchestrator;
        # recompute both modes against this identical source snapshot.
        adapter = run(prepared, source, cache, tmp_path / 'adapter_real_class',
                      dimensions=dims, device='cpu')
        official = run(prepared, source, cache, tmp_path / 'official_real_class',
                       dimensions=dims, device='cpu', mode='official')
        assert successful(official)
        assert compare(official, adapter, dims)['passed']



"""Fixed display protocol and incomplete-run semantics, without model inference."""
import copy
import csv

import pytest

from ais_bench.benchmark.vabench import DIMENSIONS
from ais_bench.benchmark.vabench.data import digest, read_json, write_json
from ais_bench.benchmark.vabench.engine import successful
from ais_bench.benchmark.vabench.results import (
    build_report, export_csv, export_official_results, format_summary,
    normalize_records, refresh_report)
from ais_bench.benchmark.vabench.scoring import RULES, normalize_score


def fixture_inputs():
    manifest = dict(input_signature='synthetic_fixture', evaluation_scope='mini',
                    records=[dict(idx='a', category='one'), dict(idx='b', category='two')])
    details = {}
    for dim in DIMENSIONS:
        low, high, _ = RULES[dim]
        score = (low + high) / 2
        raw = [dict(video_path=r['idx'] + '.mp4', video_results=score,
                    reasoning='Synthetic test fixture, not model output') for r in manifest['records']]
        details[dim] = dict(records=normalize_records(dim, raw, manifest['records']), requested_device='fixture')
    return manifest, details


@pytest.mark.parametrize('dimension', DIMENSIONS)
def test_fixed_endpoints_and_midpoints(dimension):
    low, high, reverse = RULES[dimension]
    assert normalize_score(dimension, low)['normalized_score'] == (100 if reverse else 0)
    assert normalize_score(dimension, high)['normalized_score'] == (0 if reverse else 100)
    assert normalize_score(dimension, (low + high) / 2)['normalized_score'] == 50
    assert normalize_score(dimension, None)['normalized_score'] is None
    assert normalize_score(dimension, float('nan'))['normalized_score'] is None


def test_all_fifteen_valid_means_produce_equal_weight_formal_total():
    manifest, details = fixture_inputs()
    report = build_report(manifest, details, {})
    assert report['total']['status'] == 'formal' and report['total']['score'] == 50
    assert report['total']['valid_dimensions'] == 15
    assert all(v == 1 / 15 for v in report['total']['weights'].values())
    assert successful(report)
    assert report['normalization_protocol']['official_total'] is False
    assert '正式总分: 50.00/100' in format_summary(report)
    # Change one normalized metric from 50 to 100; total rises by 50/15.
    for row in details['fourth_qa_audio']['records']:
        row['score'] = 1
    assert build_report(manifest, details, {})['total']['score'] == pytest.approx(50 + 50 / 15)


def test_lsa_omission_keeps_mean_and_prevents_formal_total():
    manifest, details = fixture_inputs()
    # An absent official record is aligned to the original input denominator.
    details['second_lsa']['records'].pop()
    report = build_report(manifest, details, {})
    item = report['metrics']['second_lsa']
    assert item['mean'] == 5 and item['coverage_percent'] == 50
    assert item['expected'] == 2 and item['scored'] == item['omitted'] == 1
    assert report['total']['status'] == 'preview' and report['total']['score'] == 50
    assert not successful(report)
    details['second_lsa']['records'].clear()
    report = build_report(manifest, details, {})
    assert report['total']['score'] is None and report['total']['status'] == 'unavailable'
    assert report['total']['valid_dimensions'] == 14


def test_single_metric_smoke_success_does_not_fabricate_a_fifteen_metric_total():
    manifest, details = fixture_inputs()
    report = build_report(manifest, {'second_clap': details['second_clap']}, {}, ['second_clap'])
    assert successful(report)
    assert len(report['metrics']) == 15 and len(report['details']) == 1
    assert report['total']['score'] is None
    assert report['metrics']['first_dnsmos']['status'] == 'not_evaluated'


def test_mean_before_clipping_and_out_of_anchor_samples():
    manifest, details = fixture_inputs()
    for row, value in zip(details['first_nisqa']['records'], [-1, 5]):
        row['score'] = value
    item = build_report(manifest, details, {})['metrics']['first_nisqa']
    assert item['mean'] == 2 and item['normalized_score'] == 25
    assert item['out_of_bounds_count'] == 1 and item['mean_out_of_bounds'] is False
    assert item['scenes']['one']['normalized_score'] == 0


def test_overall_mean_weights_videos_instead_of_scenes():
    entries = [dict(idx='a', category='one'), dict(idx='b', category='one'),
               dict(idx='c', category='two')]
    raw = [dict(video_path=r['idx'] + '.mp4', video_results=s) for r, s in zip(entries, [1, 1, 5])]
    details = {'first_nisqa': dict(records=normalize_records('first_nisqa', raw, entries))}
    report = build_report(dict(input_signature='fixture', records=entries), details, {})
    metric = report['metrics']['first_nisqa']
    assert metric['mean'] == pytest.approx(7 / 3)
    assert metric['normalized_score'] == pytest.approx(100 / 3)
    assert metric['scenes']['one']['normalized_score'] == 0
    assert metric['scenes']['two']['normalized_score'] == 100


def test_failure_scores_stay_null_and_original_fallback_stays_auditable():
    expected = [dict(idx='a', category='scene')]
    raw = [dict(video_path='a.mp4', video_results=1.0,
                reasoning='No reasoning provided (empty or OOM/parse fallback).')]
    row = normalize_records('third_alignment', raw, expected)[0]
    assert row['status'] == 'parse_error' and row['score'] is None
    assert row['original_score'] == 1 and row['raw'] == raw[0]
    qa = [dict(video_path='a.mp4', video_results=0, answers=[
        dict(answer='no', reason='OOM_ERROR')])]
    row = normalize_records('fourth_qa_audio', qa, expected)[0]
    assert row['status'] == 'backend_error' and row['score'] is None
    # Legitimate zero QA and low macro grades remain valid.
    assert normalize_records('fourth_qa_audio', [dict(video_path='a.mp4', video_results=0)], expected)[0]['status'] == 'ok'
    assert normalize_records('third_alignment', [dict(video_path='a.mp4', video_results=1,
                             reasoning='Valid low grade')], expected)[0]['status'] == 'ok'


def test_failed_sample_is_excluded_without_changing_remaining_mean():
    manifest, details = fixture_inputs()
    row = details['second_desync']['records'][0]
    row.update(status='backend_error', original_score=1.0, score=1.0)
    report = build_report(manifest, details, {})
    assert report['details']['second_desync']['records'][0]['score'] is None
    assert report['metrics']['second_desync']['mean'] == 1.0
    assert report['metrics']['second_desync']['failed'] == 1
    assert report['total']['status'] == 'preview'


def test_unattributed_backend_errors_cannot_certify_a_formal_total():
    manifest, details = fixture_inputs()
    details['second_clap']['errors'] = [{'error': 'Unattributed backend failure'}]
    report = build_report(manifest, details, {})
    assert report['metrics']['second_clap']['coverage_percent'] == 100
    assert report['metrics']['second_clap']['backend_issue_count'] == 1
    assert report['total']['status'] == 'preview'
    assert not successful(report)


def test_csv_preserves_original_means_and_adds_normalized_and_total_outputs(tmp_path):
    manifest, details = fixture_inputs()
    report = build_report(manifest, details, {})
    export_csv(report, tmp_path)
    with (tmp_path / 'metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 15 and all(float(r['normalized_score']) == 50 for r in rows)
    with (tmp_path / 'total.csv').open(encoding='utf-8-sig', newline='') as stream:
        row = next(csv.DictReader(stream))
    assert row['status'] == 'formal' and float(row['score']) == 50
    assert '正式总分' in (tmp_path / 'summary.md').read_text(encoding='utf-8')


def test_official_export_keeps_raw_scores_and_uses_official_prompt_csv(tmp_path):
    entries = [dict(idx='a', category='one'), dict(idx='b', category='one')]
    prepared = tmp_path / 'prepared'
    prompts = [
        dict(prompt_en='Prompt A', video_list=['videos/a.mp4'], audio_list=['audio/a.wav']),
        dict(prompt_en='Prompt B', video_list=['videos/b.mp4'], audio_list=['audio/b.wav']),
    ]
    write_json(prepared / 'json' / 'one.json', prompts)
    dnsmos = [dict(audio_path='audio/a.wav', video_results=1),
              dict(audio_path='audio/b.wav', video_results=3)]
    desync = [dict(video_path='videos/a.mp4', video_results=0.2),
              dict(video_path='videos/b.mp4', video_results=1.0)]
    details = {'first_dnsmos': dict(records=normalize_records('first_dnsmos', dnsmos, entries)),
               'second_desync': dict(records=normalize_records('second_desync', desync, entries)),
               'second_lsa': dict(records=normalize_records(
                   'second_lsa', [dict(video_path='videos/a.mp4', video_results=5)], entries))}
    # AISBench refuses this fallback score, but official evaluate.py reports it.
    details['second_desync']['records'][1].update(status='backend_error', score=None,
                                                  original_score=1.0)
    report = build_report(dict(input_signature='fixture', records=entries), details, {})
    export_official_results(report, tmp_path, prepared=prepared,
                            timestamp='2026-01-02-03:04:05')
    folder = tmp_path / 'official_eval_results' / 'one'
    results = read_json(folder / 'one_2026-01-02-03:04:05_eval_results.json')
    assert list(results) == ['first_dnsmos', 'second_desync', 'second_lsa']
    assert [r['video_results'] for r in results['first_dnsmos']['video_results']] == [1, 3]
    assert [r['video_results'] for r in results['second_desync']['video_results']] == [0.2, 1.0]
    assert len(results['second_lsa']['video_results']) == 1
    with (folder / 'one_2026-01-02-03:04:05_eval_results.csv').open(
            encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [r['prompt_name'] for r in rows] == ['Prompt A', 'Prompt B']
    assert [float(r['first_dnsmos']) for r in rows] == [1, 3]
    assert [float(r['second_desync']) for r in rows] == [0.2, 1.0]
    assert [float(r['second_lsa']) for r in rows] == [5, -1]
    assert read_json(folder / 'one_2026-01-02-03:04:05_full_info.json') == prompts


def test_official_export_without_prepared_falls_back_to_raw_prompt_then_idx(tmp_path):
    entries = [dict(idx='a', category='one'), dict(idx='b', category='one')]
    raw = [dict(video_path='a.mp4', video_results=0.1, prompt='Prompt A'),
           dict(video_path='b.mp4', video_results=0.3)]
    details = {'second_desync': dict(records=normalize_records('second_desync', raw, entries))}
    report = build_report(dict(input_signature='fixture', records=entries), details, {})
    export_official_results(report, tmp_path, timestamp='fixture')
    folder = tmp_path / 'official_eval_results' / 'one'
    with (folder / 'one_fixture_eval_results.csv').open(
            encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [r['prompt_name'] for r in rows] == ['Prompt A', 'b']
    assert [float(r['second_desync']) for r in rows] == [0.1, 0.3]
    assert not (folder / 'one_fixture_full_info.json').exists()


def test_summarize_cli_reuses_old_raw_records_and_does_not_modify_input(tmp_path, capsys):
    from ais_bench.benchmark.vabench.cli import main
    manifest, details = fixture_inputs()
    original = build_report(manifest, details, {})
    original['schema_version'] = 1
    original.pop('total')
    source = tmp_path / 'old_report.json'
    write_json(source, original)
    signature = digest(source)
    output = tmp_path / 'new_summary'
    assert main(['summarize', '--report', str(source), '--output', str(output)]) == 0
    report = read_json(output / 'report.json')
    assert report['total']['score'] == 50 and digest(source) == signature
    assert report['postprocessing']['model_inference_rerun'] is False
    assert report['postprocessing']['source_report_sha256'] == signature
    assert '正式总分' in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        main(['summarize', '--report', str(source), '--output', str(output)])


def test_summarize_cli_restores_official_prompt_names_from_prepared(tmp_path):
    from ais_bench.benchmark.vabench.cli import main
    entries = [dict(idx='a', category='one')]
    raw = [dict(video_path='videos/a.mp4', video_results=0.2)]
    details = {'second_desync': dict(records=normalize_records('second_desync', raw, entries))}
    report = build_report(dict(input_signature='fixture', records=entries), details, {})
    source = tmp_path / 'report.json'
    write_json(source, report)
    prepared = tmp_path / 'prepared'
    write_json(prepared / 'json' / 'one.json',
               [dict(prompt_en='Prompt A', video_list=['videos/a.mp4'], audio_list=[])])
    output = tmp_path / 'summary'
    assert main(['summarize', '--report', str(source), '--output', str(output),
                 '--prepared', str(prepared)]) == 0
    path = next(output.glob('official_eval_results/*/*_eval_results.csv'))
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['prompt_name'] == 'Prompt A'
    assert float(rows[0]['second_desync']) == 0.2


def test_old_false_success_is_reclassified_without_mutating_original_report():
    manifest, details = fixture_inputs()
    old = build_report(manifest, details, {})
    old['schema_version'] = 1
    row = old['details']['third_alignment']['records'][0]
    row.update(status='ok', score=1.0)
    row.pop('original_score')
    row['raw'].update(video_results=1.0,
                      reasoning='No reasoning provided (empty or OOM/parse fallback).')
    snapshot = copy.deepcopy(old)
    refreshed = refresh_report(old)
    assert old == snapshot
    row = refreshed['details']['third_alignment']['records'][0]
    assert row['score'] is None and row['original_score'] == 1.0
    assert row['status'] == 'parse_error'
    assert refreshed['total']['status'] == 'preview'


def test_aisbench_table_shows_fifteen_metrics_and_one_total(tmp_path):
    from mmengine.config import ConfigDict
    from ais_bench.benchmark.summarizers.vabench import VABenchSummarizer
    manifest, details = fixture_inputs()
    report = build_report(manifest, details, {})
    write_json(tmp_path / 'results/test/vabench.json', report)
    summarizer = VABenchSummarizer(ConfigDict(models=[dict(abbr='test')],
        datasets=[dict(abbr='vabench', path='fixture')], work_dir=str(tmp_path)))
    raw, parsed, metrics, modes = summarizer._pick_up_results()
    assert len(parsed['test']) == 16
    assert parsed['test']['vabench/total']['normalized_score'] == 50
    assert parsed['test']['vabench/total']['status'] == '正式总分'
    table = summarizer._format_table(parsed, metrics, modes)
    assert len(table) == 17 and table[-1][4] == '50.00'
    summarizer.summarize(output_path=str(tmp_path / 'summary/fixture.txt'), time_str='fixture')
    for name in ('fixture.csv', 'fixture.md'):
        text = (tmp_path / 'summary' / name).read_text(encoding='utf-8')
        assert 'vabench/total' in text and '正式总分' in text


def test_duplicate_and_unexpected_sample_records_cannot_certify_total():
    manifest, details = fixture_inputs()
    details['first_dnsmos']['records'].append(copy.deepcopy(details['first_dnsmos']['records'][0]))
    with pytest.raises(ValueError, match='duplicate/unexpected'):
        build_report(manifest, details, {})
