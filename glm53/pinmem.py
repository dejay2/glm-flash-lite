"""Page-aligned, GPU-mapped host memory whose UVA device alias equals the host address.

Every host region the NVMe/RAM tiers share with the GPU (control block, RAM-slot arenas, pf rings, zero-copy arena,
host embedding) must satisfy pinned_cuda_view(p).data_ptr() == host address: the CPU lane, the O_DIRECT preads and the
GPU pointer tables all use one number for both sides.

  register   anon mmap + cudaHostRegister(Mapped)            native Linux (alias == host address)
  hostalloc  cudaHostAlloc(Portable|Mapped)                  WSL2: cudaHostRegister returns a *different* device pointer
                                                             for mmap memory, cudaHostAlloc does not
Env GLM53_PIN_ALLOC=auto|register|hostalloc (default auto: probe once with a one-page mmap + register; use register if
the alias equals the host address, else hostalloc). Memory from cudaHostAlloc is freed with cudaFreeHost when the
returned buffer object is garbage collected (nothing munmaps/madvises these regions).
"""
import ctypes, mmap, os
import torch

PORTABLE, MAPPED = 1, 2          # cudaHostAllocPortable / cudaHostAllocMapped (MAPPED == cudaHostRegisterMapped)
_MODE = None
_CUDART = None


def _cudart():
    global _CUDART
    if _CUDART is None:
        last = None
        for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
            try:
                _CUDART = ctypes.CDLL(name)
                break
            except OSError as e:
                last = e
        if _CUDART is None:
            raise RuntimeError(f"pinmem: cannot load libcudart for cudaHostAlloc: {last}")
        _CUDART.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        _CUDART.cudaFreeHost.argtypes = [ctypes.c_void_p]
        _CUDART.cudaGetErrorString.restype = ctypes.c_char_p
        _CUDART.cudaGetErrorString.argtypes = [ctypes.c_int]
    return _CUDART


def _err(code):
    return f"{int(code)} ({_cudart().cudaGetErrorString(int(code)).decode()})"


def _register(nbytes, dev_idx):
    """anon mmap + cudaHostRegister(Mapped). Returns (mm, addr)."""
    mm = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    a = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    with torch.cuda.device(dev_idx):
        r = torch.cuda.cudart().cudaHostRegister(a, nbytes, MAPPED)
    if int(r) != 0:
        raise RuntimeError(f"cudaHostRegister failed ({nbytes} bytes = {nbytes / 2**30:.2f} GiB): {r}")
    return mm, a


def _alias(buf, dev_idx):
    from exllamav3.ext import exllamav3_ext as xe
    t = torch.frombuffer(buf, dtype=torch.uint8, count=4096)
    return xe.pinned_cuda_view(t, dev_idx).data_ptr()


def mode(dev_idx=0):
    global _MODE
    if _MODE is None:
        want = os.environ.get("GLM53_PIN_ALLOC", "auto").lower()
        if want not in ("auto", "register", "hostalloc"):
            raise ValueError(f"GLM53_PIN_ALLOC={want!r}: expected auto|register|hostalloc")
        if want == "auto":
            mm, a = _register(4096, dev_idx)
            al = _alias(mm, dev_idx)
            with torch.cuda.device(dev_idx):
                r = torch.cuda.cudart().cudaHostUnregister(a)
            if int(r) != 0:
                raise RuntimeError(f"pinmem: cudaHostUnregister of the probe page failed: {r}")
            want = "register" if al == a else "hostalloc"
            print(f" -- pinmem: GLM53_PIN_ALLOC=auto -> {want} (probe: host 0x{a:x}, device alias 0x{al:x})", flush=True)
        _MODE = want
    return _MODE


def _hostalloc(nbytes, dev_idx):
    cu = _cudart()
    p = ctypes.c_void_p()
    with torch.cuda.device(dev_idx):
        torch.cuda.current_stream(dev_idx)       # make sure the primary context exists on this device
        raw = nbytes + 4096                      # small cudaHostAlloc blocks are sub-allocated, not page aligned: slack
        r = cu.cudaHostAlloc(ctypes.byref(p), raw, PORTABLE | MAPPED)
    if r != 0 or not p.value:
        raise RuntimeError(f"cudaHostAlloc({raw} bytes = {nbytes / 2**30:.2f} GiB, Portable|Mapped) failed: {_err(r)}")
    base = p.value
    addr = (base + 4095) // 4096 * 4096          # first page-aligned address inside the block (alias == host still holds)

    class HostAllocBuf(ctypes.c_ubyte * nbytes):
        def __del__(self, _cu=cu, _a=base):
            _cu.cudaFreeHost(_a)   # frees the raw block; argtypes c_void_p; no module globals (runs at shutdown)

    return HostAllocBuf.from_address(addr), addr


def alloc(nbytes, dev_idx, tag=""):
    """Returns (buf, addr): buf supports the buffer protocol (torch.frombuffer / ctypes from_buffer) and owns the memory
    (keep it referenced); addr is the page-aligned host address == UVA device alias (asserted)."""
    nbytes = (nbytes + 4095) // 4096 * 4096
    m = mode(dev_idx)
    buf, a = _register(nbytes, dev_idx) if m == "register" else _hostalloc(nbytes, dev_idx)
    assert a % 4096 == 0, f"pinmem: {tag} host address 0x{a:x} not page aligned"
    al = _alias(buf, dev_idx)
    assert al == a, f"UVA alias != host address ({tag}, mode={m}): host 0x{a:x} device 0x{al:x}"
    return buf, a
