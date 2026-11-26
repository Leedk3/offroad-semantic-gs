#!/usr/bin/env bash
set -euo pipefail

# The upstream renderer and these patch hunks retain their separate research /
# noncommercial license. This script does not change those terms.
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_destination="$task_root/.external/diff-surfel-rasterization"
task_python="${OFFROAD_PYTHON:-python}"
task_accept=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --accept-research-license) task_accept=true; shift ;;
    --destination) task_destination="$2"; shift 2 ;;
    --help)
      echo 'Usage: bash scripts/install_native.sh --accept-research-license [--destination DIRECTORY]'
      echo 'Read licenses/Gaussian-Splatting.txt first. Requires PyTorch, CUDA toolkit, nvcc, git and a C++ compiler.'
      echo 'Set OFFROAD_PYTHON to the Python environment that will build and use the extension.'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$task_accept" != true ]]; then
  echo 'Read licenses/Gaussian-Splatting.txt and explicitly pass --accept-research-license to install.' >&2
  exit 2
fi
if [[ -e "$task_destination" ]]; then
  echo "Destination exists; choose a new directory to avoid changing an existing checkout: $task_destination" >&2
  exit 2
fi
git clone --no-checkout https://github.com/hbb1/diff-surfel-rasterization.git "$task_destination"
git -C "$task_destination" checkout --detach e0ed0207b3e0669960cfad70852200a4a5847f61
git -C "$task_destination" submodule update --init --recursive
task_glm="$(git -C "$task_destination/third_party/glm" rev-parse HEAD)"
[[ "$task_glm" == 5c46b9c07008ae65cb81ab79cd677ecc1934b903 ]]
git -C "$task_destination" apply "$task_root/patches/2dgs_conditional_center_forward.patch"
git -C "$task_destination" apply "$task_root/patches/2dgs_center_depth_backward_consistency.patch"
"$task_python" - "$task_destination" "$task_root/native_runtime.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); specification=json.loads(Path(sys.argv[2]).read_text())
for name,expected in specification['patched_source_SHA256'].items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual!=expected:raise SystemExit('Patched source hash mismatch: '+name)
print('Pinned renderer, GLM and patched sources verified.')
PY
"$task_python" -m pip install --no-build-isolation "$task_destination"
"$task_python" - "$task_destination" "$task_root/native_runtime.json" <<'PY'
import hashlib,importlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); specification=json.loads(Path(sys.argv[2]).read_text())
renderer=importlib.import_module('diff_surfel_rasterization')
extension=importlib.import_module('diff_surfel_rasterization._C')
marker={key:specification[key] for key in ['renderer_commit','glm_commit','patched_source_SHA256']}
marker['compiled_extension_SHA256']=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()
marker['schema_version']=1
path=Path(renderer.__file__).parent/'offroad_runtime.json'
path.write_text(json.dumps(marker,indent=2)+'\n')
print('Installed renderer and recorded the compiled extension hash.')
PY
