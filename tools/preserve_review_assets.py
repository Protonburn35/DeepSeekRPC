"""Package only the actual matching Windows-qualified source and artifacts."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

repo = Path(os.environ['DSR_CANDIDATE_REPO'])
base = Path(os.environ['DSR_EVIDENCE_ROOT'])
commit = '25de683e5ef3a9299267b8e61bafb1d755941c84'
runtime = base / 'runtime-25de683'
cpu = base / 'windows-cpu-25de683'
packet = base / 'review-25de683'
assets = base / 'release-25de683'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == commit
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


if '--verify-upload' in sys.argv:
    result = json.loads(subprocess.check_output(['gh', 'api',
        'repos/Protonburn35/DeepSeekRPC/releases/' + os.environ['DSR_RELEASE_ID']], text=True))
    assert result['draft'] is True and result['tag_name'] == 'whole-source-25de683'
    remote = {item['name']: item for item in result['assets']}
    names = ['DeepSeekRPC-Astra-review-25de683.zip', 'DeepSeekRPC-Windows-runtime-25de683.zip',
        'DeepSeekRPC-native-components-25de683.zip', 'CHECKS.json', 'SHA256SUMS.txt']
    records = {}
    for name in names:
        path = assets / name
        item = remote[name]
        expected = digest(path)
        assert item['state'] == 'uploaded' and item['size'] == path.stat().st_size
        assert item['digest'] == 'sha256:' + expected
        records[name] = {'bytes': item['size'], 'sha256': expected}
    report = {'source_commit': commit, 'draft_release': result['html_url'],
        'status': 'all five actual uploaded asset lengths and GitHub SHA-256 values match',
        'assets': records, 'GPU_execution': 'not run', 'independent_review': 'follow-up pending; first review R1/R2/R3 repaired'}
    (assets / 'UPLOAD.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0)


manifest = json.loads((runtime / 'BUNDLE.json').read_text())
assert manifest['source_commit'] == commit
seen = set()
for name, record in manifest['files'].items():
    relative = PurePosixPath(name)
    assert not relative.is_absolute() and '..' not in relative.parts and '\\' not in name
    assert name.casefold() not in seen
    seen.add(name.casefold())
    path = runtime.joinpath(*relative.parts)
    assert not path.is_symlink() and path.resolve(strict=True).is_relative_to(runtime)
    assert path.stat().st_size == record['bytes'] and digest(path) == record['sha256'], name
assert {p.relative_to(runtime).as_posix() for p in runtime.rglob('*') if p.is_file()} == set(manifest['files']) | {'BUNDLE.json'}
sys.path.insert(0, str(repo / 'src'))
from deepseekrpc.runtime_identity import application_source_identity
assert application_source_identity() == manifest['application_source_identity']
sys.path.pop(0)
suite, = ET.parse(cpu / 'pytest.xml').getroot().iter('testsuite')
assert {k: int(suite.get(k)) for k in ('tests', 'failures', 'errors', 'skipped')} == {'tests': 405, 'failures': 0, 'errors': 0, 'skipped': 1}
sys.path.insert(0, str(repo / 'tools'))
from build_review_packet import validated_evidence
from build_bundle import report_wheel
_, selected = validated_evidence(runtime.resolve(), cpu / 'BUILD.json', commit)
parser = selected['evidence/installed-runtime/parser.json']
assert len(parser['checks']) == 20 and all(item['status'] == 'passed' for item in parser['checks'])
subprocess.run([sys.executable, str(repo / 'tools/build_review_packet.py'), '--runtime', str(runtime),
    '--cpu-report', str(cpu / 'BUILD.json'), '--output', str(packet)], cwd=repo, check=True)
assets.mkdir(exist_ok=False)
components = base / 'native-components-25de683'
components.mkdir(exist_ok=False)
runs = {'vllm': 39428534610, 'exllama': 39345608258, 'native': 39410211028}
native_records = {}
for name, run in runs.items():
    directory = components / name
    (directory / 'wheels').mkdir(parents=True)
    report = json.loads((runtime / f'evidence/{name}.json').read_text())
    shutil.copyfile(runtime / f'evidence/{name}.json', directory / 'BUILD.json')
    wheel = runtime / 'wheels' / report['wheel']['name']
    os.link(wheel, directory / 'wheels' / wheel.name)
    admitted, actual = report_wheel(directory, component=name)
    assert digest(admitted) == actual['wheel']['sha256']
    native_records[name] = {'run': run, 'source_commit': report['source_commit'], 'wheel': report['wheel']}
for name in ('licenses', 'provenance', 'patches'):
    shutil.copytree(runtime / name, components / name)
for name in ('LICENSE', 'THIRD_PARTY_NOTICES.md', 'SOURCE.md'):
    shutil.copyfile(runtime / name, components / name)
(components / 'COMPONENTS.json').write_text(json.dumps({'kind': 'retained actual qualified native components',
    'application_source_commit': commit, 'components': native_records, 'GPU_execution': 'not run'}, indent=2) + '\n')


def archive_tree(directory, name):
    target = assets / name
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(directory.rglob('*')):
            assert not path.is_symlink()
            if path.is_file():
                archive.write(path, path.relative_to(directory).as_posix())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
    assert target.stat().st_size < 2 * 1024 ** 3
    return {'name': name, 'bytes': target.stat().st_size, 'sha256': digest(target)}


records = [archive_tree(packet, 'DeepSeekRPC-Astra-review-25de683.zip'),
    archive_tree(runtime, 'DeepSeekRPC-Windows-runtime-25de683.zip'),
    archive_tree(components, 'DeepSeekRPC-native-components-25de683.zip')]
summary = {'source_commit': commit, 'Windows_CPU': {'run': 39481164697, 'passed': 404, 'skipped': 1},
    'Windows_installed_runtime': {'run': 39481178727, 'checks': 8, 'original_parser_cases': 20},
    'application_source_identity': manifest['application_source_identity'],
    'runtime_files_verified': len(manifest['files']), 'native_components': native_records,
    'GPU_execution': 'not run', 'full_model_validation': 'not run', 'physical_residency': 'not run',
    'performance': 'not measured', 'independent_review': 'follow-up pending; first review R1/R2/R3 repaired', 'release_candidate': False, 'archives': records}
(assets / 'CHECKS.json').write_text(json.dumps(summary, indent=2) + '\n')
checksums = [*records, {'name': 'CHECKS.json', 'sha256': digest(assets / 'CHECKS.json')}]
(assets / 'SHA256SUMS.txt').write_text(''.join(f"{item['sha256']}  {item['name']}\n" for item in checksums))
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo)
print(json.dumps(summary, indent=2), flush=True)
