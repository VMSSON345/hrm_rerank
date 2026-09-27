"""Start the installed local MinerU API if the configured endpoint is offline."""
import os
from pathlib import Path
import subprocess
import time
import urllib.parse
import urllib.request


def ensure_local_mineru(root: Path, endpoint: str):
    try:
        with urllib.request.urlopen(endpoint.rstrip('/') + '/health', timeout=3):
            return None
    except OSError:
        pass
    url = urllib.parse.urlsplit(endpoint)
    if url.hostname not in {'127.0.0.1', 'localhost', '::1'}:
        print('MinerU endpoint unavailable: ' + endpoint, flush=True)
        return None
    executable = root / '.venv-mineru' / ('Scripts/mineru-api.exe' if os.name == 'nt' else 'bin/mineru-api')
    if not executable.is_file():
        print('MinerU is not installed in this environment. Run scripts/setup_mineru_venv.sh in WSL, then scripts/run_mineru_api.sh.', flush=True)
        return None
    env = os.environ.copy()
    # ML environments must not inherit the HRM Python import path.
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.setdefault('MINERU_API_OUTPUT_ROOT', str(root / 'cache/mineru-api'))
    env.setdefault('MINERU_PROCESSING_WINDOW_SIZE', '2')
    env.setdefault('MINERU_API_MAX_CONCURRENT_REQUESTS', '1')
    print('Starting installed MinerU API: ' + endpoint, flush=True)
    child = subprocess.Popen([str(executable), '--host', url.hostname, '--port', str(url.port or 8000), '--max-concurrency', '1'], cwd=root, env=env)

    for _ in range(60):
        if child.poll() is not None:
            print('MinerU exited during startup; check its log.', flush=True)
            return child
        try:
            with urllib.request.urlopen(endpoint.rstrip('/') + '/health', timeout=1):
                return child
        except OSError:
            time.sleep(1)
    print('MinerU is still starting. Wait for its health endpoint before uploading.', flush=True)
    return child
