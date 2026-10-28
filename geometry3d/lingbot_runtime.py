"""Invoke the LingBot adapter in a separate local environment or SSH Docker job.

Upstream: https://github.com/Robbyant/lingbot-map
Clone/install that project in its own Python environment using its upstream
instructions, and download its lingbot-map-long.pt checkpoint separately.
Our repository contains the adapter, not LingBot or its weights. Start with
lingbot.local.example.json and replace its three absolute-path placeholders.

--lingbot-config JSON examples (paths are configuration, not repository code):
 {"runtime":"local", "python":"/venv/bin/python", "repo":"/repos/lingbot-map",
  "checkpoint":"/models/lingbot-map-long.pt", "device":"cuda"}
 {"runtime":"ssh-docker", "host":"user@host", "port":2222,
  "identity_file":"~/.ssh/id_ed25519", "repo":"/srv/projects/lingbot-map",
  "checkpoint":"/srv/projects/lingbot-map/data/models/lingbot-map-long.pt",
  "image":"lingbot-map:cuda", "remote_workdir":"/srv/projects/lingbot-map/data/pipeline_jobs"}

The remote checkout is mounted read-only. Only this adapter and the exact clip
are uploaded. Depth exports are downloaded before local geometry processing.
Remote jobs are retained for diagnostics; no system packages/drivers are changed.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import uuid

ADAPTER = Path(__file__).with_name('lingbot_inference.py')
UPSTREAM_REPOSITORY = 'https://github.com/Robbyant/lingbot-map'


def required(config, key):
    value = config.get(key)
    if not isinstance(value, str) or not value.strip() or value.startswith('-'):
        raise ValueError(f'LingBot runtime config requires {key}')
    return value


def adapter_args(config, repo, checkpoint, video, output):
    return ['--repo', str(repo), '--checkpoint', str(checkpoint), '--video', str(video),
            '--output', str(output), '--device', str(config.get('device', 'cuda')),
            '--dtype', str(config.get('dtype', 'auto')),
            '--scale-frames', str(config.get('scale_frames', 8)),
            '--cache-window', str(config.get('cache_window', 64)),
            '--keyframe-interval', str(config.get('keyframe_interval', 1))]


def local_path(value, base):
    p = Path(value).expanduser()
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


def run(config, config_path, video, output, log):
    runtime = config.get('runtime')
    base = Path(config_path).resolve().parent
    video, output = Path(video).resolve(), Path(output).resolve()
    if runtime == 'local':
        python = local_path(required(config, 'python'), base)
        repo = local_path(required(config, 'repo'), base)
        checkpoint = local_path(required(config, 'checkpoint'), base)
        if not python.is_file():
            raise FileNotFoundError(f'LingBot Python executable missing: {python}. '
                f'Install {UPSTREAM_REPOSITORY} in its own environment and set python, repo and checkpoint '
                'in --lingbot-config (see lingbot.local.example.json).')
        argv = [str(python), str(ADAPTER), *adapter_args(config, repo, checkpoint, video, output)]
        subprocess.run(argv, cwd=repo, check=True, stdout=log, stderr=subprocess.STDOUT)
        return
    if runtime != 'ssh-docker':
        raise ValueError('LingBot runtime must be local or ssh-docker')
    host = required(config, 'host')
    if any(c.isspace() for c in host):
        raise ValueError('SSH host must not contain whitespace')
    port = int(config.get('port', 22))
    if not 1 <= port <= 65535:
        raise ValueError('Invalid SSH port')
    identity = ['-i', str(local_path(config['identity_file'], base))] if config.get('identity_file') else []
    options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', *identity]
    ssh = ['ssh', '-p', str(port), *options, host]
    scp = ['scp', '-P', str(port), *options]
    user, at, address = host.rpartition('@')
    address = address or host
    # scp requires brackets for IPv6, whereas ssh accepts the plain address.
    if ':' in address and not address.startswith('['):
        address = '[' + address + ']'
    scp_host = (user + at if at else '') + address
    repo, checkpoint, root = (PurePosixPath(required(config, k)) for k in ('repo', 'checkpoint', 'remote_workdir'))
    if not all(p.is_absolute() for p in (repo, checkpoint, root)):
        raise ValueError('Remote repo, checkpoint and workdir must be absolute POSIX paths')
    image = required(config, 'image')
    job = root / ('job-' + uuid.uuid4().hex)

    def remote(argv, capture=False):
        result = subprocess.run([*ssh, shlex.join([str(a) for a in argv])], check=True,
            stdout=subprocess.PIPE if capture else log, stderr=log, text=True)
        return result.stdout.strip() if capture else None

    # Fail before transferring video when the server GPU runtime is broken.
    if str(config.get('device', 'cuda')).startswith('cuda'):
        remote(['nvidia-smi', '-L'])
    uid, gid = remote(['id', '-u'], capture=True), remote(['id', '-g'], capture=True)
    if not uid.isdecimal() or not gid.isdecimal():
        raise ValueError('Cannot determine remote Docker user')
    remote(['mkdir', '-p', str(job)])
    subprocess.run([*scp, str(ADAPTER), f'{scp_host}:{job}/lingbot_inference.py'],
                   check=True, stdout=log, stderr=subprocess.STDOUT)
    subprocess.run([*scp, str(video), f'{scp_host}:{job}/{video.name}'],
                   check=True, stdout=log, stderr=subprocess.STDOUT)
    command = ['docker', 'run', '--rm', '--shm-size=8g', '--user', f'{uid}:{gid}',
               '-e', 'HOME=/tmp', '-e', 'PYTHONDONTWRITEBYTECODE=1',
               '--mount', f'type=bind,src={repo},dst=/lingbot,readonly',
               '--mount', f'type=bind,src={checkpoint},dst=/model/weights.pt,readonly',
               '--mount', f'type=bind,src={job},dst=/job', '-w', '/lingbot']
    if str(config.get('device', 'cuda')).startswith('cuda'):
        command.extend(['--gpus', 'all'])
    command.extend([image, 'python', '/job/lingbot_inference.py',
                    *adapter_args(config, '/lingbot', '/model/weights.pt', '/job/' + video.name, '/job/output')])
    remote(command)
    # Download into an empty staging folder; never accept a partial scp as a run.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='.download-', dir=output) as temp:
        subprocess.run([*scp, '-r', f'{scp_host}:{job}/output', temp],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
        downloaded = Path(temp) / 'output'
        manifest = downloaded / 'processing_manifest.json'
        if not manifest.is_file() or not (downloaded / video.stem).is_dir():
            raise ValueError('Remote LingBot run did not export a complete result')
        data = json.loads(manifest.read_text(encoding='utf-8'))
        data['remote_execution'] = {'host': host, 'job': str(job), 'image': image,
                                    'repo': str(repo), 'checkpoint': str(checkpoint)}
        manifest.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        if (output / video.stem).exists() or (output / manifest.name).exists():
            raise FileExistsError('Refusing to overwrite downloaded LingBot predictions')
        (downloaded / video.stem).rename(output / video.stem)
        manifest.rename(output / manifest.name)
