#!/usr/bin/env python3
"""GPU unit test for glm53/pinmem.py. Needs a GPU and a store file; run inside the image:
  docker run --rm --gpus '"device=0"' --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
    -v /mnt/wsl/glmstore/nvx:/nvx:ro -e GLM53_PIN_ALLOC=auto glm53-5090:dev python3 /opt/glm53/tests/test_pinmem.py
Checks: alias == host address, O_DIRECT pread into the region, GPU read through the alias == file bytes.
GLM53_PIN_ALLOC=register on WSL2 must FAIL the alias assert (the bug this fixes)."""
import ctypes, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glm53"))
import torch
import pinmem
from exllamav3.ext import exllamav3_ext as xe

STORE = os.environ.get("GLM53_TEST_FILE", "/nvx/glm53_flash_exl3_3.05bpw_experts.bin")
N = 64 << 20      # 64 MiB region
OFF = 4 << 20     # file offset (4096 aligned)
dev = 0
torch.zeros(1, device=f"cuda:{dev}")
buf, a = pinmem.alloc(N, dev, "test")
print(f"mode={pinmem.mode(dev)} host=0x{a:x}")
t = torch.frombuffer(buf, dtype=torch.uint8, count=N)
alias = xe.pinned_cuda_view(t, dev)
assert alias.data_ptr() == a, "alias != host"
libc = ctypes.CDLL(None, use_errno=True)
fd = os.open(STORE, os.O_RDONLY | os.O_DIRECT)
n = libc.pread(fd, ctypes.c_void_p(a), ctypes.c_size_t(N), ctypes.c_long(OFF))
assert n == N, f"pread returned {n} errno {ctypes.get_errno()}"
ref = os.pread(os.open(STORE, os.O_RDONLY), N, OFF)
assert bytes(t.numpy()) == ref, "host bytes != file bytes"
out = torch.empty(N, dtype=torch.uint8, device=f"cuda:{dev}")
out.copy_(alias[: N])              # GPU reads the host bytes through the alias pointer
torch.cuda.synchronize()
assert bytes(out.cpu().numpy()) == ref, "GPU bytes via alias != file bytes"
print("PASS: alias==host, O_DIRECT pread ok, GPU copy via alias matches file")
