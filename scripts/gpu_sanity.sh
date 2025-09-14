#!/usr/bin/env bash
# gpu_sanity.sh — Verify NVIDIA driver + Container Toolkit for GPU containers
set -u

RED=\033[31m
GRN=\033[32m
YEL=\033[33m
NC=\033[0m

pass() { echo -e "${GRN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
warn() { echo -e "${YEL}[WARN]${NC} $*"; }

echo "== GPU Sanity Check =="

overall_ok=1

# 1) Host driver present
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    pass "Host nvidia-smi works"
  else
    fail "Host nvidia-smi failed. Install/repair NVIDIA driver and reboot."
    overall_ok=0
  fi
else
  fail "nvidia-smi not found. Install NVIDIA driver."
  overall_ok=0
fi

# 2) Kernel modules loaded
if lsmod | grep -q '^nvidia'; then
  pass "Kernel modules: nvidia loaded"
else
  warn "Kernel modules: nvidia not listed. Secure Boot or driver issue?"
fi

# 3) Device nodes exist
if ls /dev/nvidiactl /dev/nvidia0 >/dev/null 2>&1; then
  pass "Device nodes present (/dev/nvidiactl, /dev/nvidia0)"
else
  warn "Missing /dev/nvidia* device nodes"
fi

# 4) NVML runtime library present (not stub)
if ldconfig -p | grep -q libnvidia-ml.so.1; then
  path=$(ldconfig -p | awk '/libnvidia-ml.so.1/ {print $NF; exit}')
  if [[ -n "${path}" ]]; then
    # If it resolves to CUDA toolkit path, it's often the stub — warn user
    if [[ "$path" == /usr/local/cuda/* ]]; then
      warn "libnvidia-ml.so.1 resolved to $path (likely CUDA stub). Prefer driver runtime at /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
      warn "Fix: sudo apt-get install -y libnvidia-ml1 nvidia-utils-<driver-major>; sudo ldconfig"
    else
      pass "libnvidia-ml.so.1 found at ${path}"
    fi
  else
    warn "libnvidia-ml.so.1 not resolved by ldconfig"
  fi
else
  fail "libnvidia-ml.so.1 not found. Install libnvidia-ml1 matching your driver."
  overall_ok=0
fi

# 5) Docker default runtime
if command -v docker >/dev/null 2>&1; then
  di=$(docker info 2>/dev/null)
  if echo "$di" | grep -qi "Default Runtime: nvidia"; then
    pass "Docker default runtime: nvidia"
  else
    warn "Docker default runtime is not 'nvidia'. Run: sudo nvidia-ctk runtime configure --runtime=docker --set-as-default && sudo systemctl restart docker"
  fi
  if echo "$di" | grep -qi rootless; then
    warn "Docker appears rootless; GPU containers are better supported with rootful Docker."
  fi
else
  fail "docker not found. Install Docker Engine."
  overall_ok=0
fi

# 6) Container NVML test
echo "Running CUDA test container: nvidia-smi"
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  pass "Container nvidia-smi works"
else
  fail "Container nvidia-smi failed. Common causes: toolkit not installed, driver/toolkit mismatch, or stub libnvidia-ml."
  echo "Hints:"
  echo "- Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
  echo "- Then: sudo nvidia-ctk runtime configure --runtime=docker --set-as-default && sudo systemctl restart docker"
  echo "- Ensure libnvidia-ml.so.1 exists and is not a stub; ldconfig -p | grep libnvidia-ml"
  overall_ok=0
fi

echo "== Summary =="
if [[ "$overall_ok" -eq 1 ]]; then
  pass "All critical checks passed. GPU containers should work."
  exit 0
else
  fail "One or more checks failed. Address the above items and re-run this script."
  exit 1
fi
