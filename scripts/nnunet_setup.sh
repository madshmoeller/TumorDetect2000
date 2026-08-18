#!/usr/bin/env bash
# Create an ISOLATED venv and install nnunetv2 into it.
#
# Deliberately NOT installed into the anaconda base environment. nnunetv2 pins
# its own torch, and letting pip resolve that in base could replace the
# 2.4.0+cu121 the pre-registered results were produced with — corrupting the very
# baseline the nnU-Net arm exists to be compared against. The venv is created
# with --system-site-packages so it can reuse the existing CUDA-enabled torch
# rather than downloading a second multi-gigabyte copy, and the install is run
# with --no-deps-on-torch semantics (torch pinned to what is already present).
set -u
ROOT=/home/mads/tumordetect
VENV=$ROOT/.venv-nnunet

echo "python: $(python3 --version)"
echo "existing torch: $(python3 -c 'import torch;print(torch.__version__, torch.version.cuda)')"

# The readiness check must import what training actually imports, not just the
# top-level package. `import nnunetv2` succeeds even when the venv is broken —
# the real failure surfaces only when the trainer pulls in skimage.
venv_ok() {
  [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c \
    "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer" 2>/dev/null
}

if venv_ok; then
  echo "venv already present and importable — nothing to do"
  exit 0
fi
rm -rf "$VENV"

python3 -m venv --system-site-packages "$VENV" || { echo "venv creation failed"; exit 1; }

# Pin the ABI-critical packages to whatever the BASE environment already has.
#
# --system-site-packages means the venv inherits base's compiled C extensions
# (skimage, and others). Those were built against base's numpy. If pip upgrades
# numpy inside the venv, every inherited C extension breaks with
# "numpy.dtype size changed ... Expected 96 from C header, got 88" — which is
# exactly what an unpinned install did here: it pulled numpy 2.2.6 against a
# base built for 1.26.4, and nnU-Net failed the moment it imported skimage.
# Pinning numpy/scipy/torch to the base versions keeps the inherited half of the
# environment binary-compatible with the installed half.
cat > /tmp/nnunet-constraints.txt <<EOF
torch==$(python3 -c 'import torch;print(torch.__version__.split("+")[0])')
numpy==$(python3 -c 'import numpy;print(numpy.__version__)')
scipy==$(python3 -c 'import scipy;print(scipy.__version__)')
EOF
echo "constraints (matched to the base env's ABI):"; sed 's/^/  /' /tmp/nnunet-constraints.txt

"$VENV/bin/pip" install --quiet --upgrade pip
if ! "$VENV/bin/pip" install --quiet -c /tmp/nnunet-constraints.txt nnunetv2; then
  echo "constrained install failed; retrying without constraint but WITHOUT upgrading torch"
  "$VENV/bin/pip" install --quiet --no-deps nnunetv2 || { echo "nnunetv2 install failed"; exit 1; }
  # nnunetv2's own runtime deps, minus torch.
  "$VENV/bin/pip" install --quiet acvl-utils dynamic-network-architectures batchgenerators \
      SimpleITK nibabel scipy scikit-image scikit-learn pandas tqdm graphviz \
      connected-components-3d batchgeneratorsv2 einops || echo "some deps failed; may still run"
fi

echo "verifying (imports what training actually imports, not just the package):"
"$VENV/bin/python" - <<'PY'
import numpy, scipy, torch
print("  numpy", numpy.__version__, " scipy", scipy.__version__)
print("  torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not visible inside the venv"
import skimage                      # the package the numpy-2 ABI break killed
print("  skimage", skimage.__version__, "OK")
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_250epochs)
print("  nnUNetTrainer + nnUNetTrainer_250epochs import OK")
PY
rc=$?
[ $rc -ne 0 ] && { echo "VERIFICATION FAILED — venv is not usable"; exit 1; }
echo "nnunet venv ready at $VENV"
