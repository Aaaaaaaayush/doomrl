"""
verify_gpu.py — Run this after activating .venv to confirm GPU is working.

What this script does:
  1. Checks PyTorch version and CUDA availability
  2. Creates a small test tensor on the GPU
  3. Confirms the RTX 5080 (SM 1.2 Blackwell) is being used
  4. Reports GPU memory available for training

Run with:
    .venv\Scripts\python.exe verify_gpu.py
"""

import sys


def main():
    print("=" * 55)
    print("  DoomRL Agent - GPU Verification")
    print("=" * 55)

    # ── 1. Check Python version ──────────────────────────────
    print(f"\n[1] Python Version : {sys.version.split()[0]}")

    # ── 2. Import PyTorch ────────────────────────────────────
    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch is not installed in this venv.")
        print("  Run: pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128")
        sys.exit(1)

    print(f"[2] PyTorch Version: {torch.__version__}")

    # ── 3. Check CUDA ────────────────────────────────────────
    cuda_available = torch.cuda.is_available()
    print(f"[3] CUDA Available  : {cuda_available}")

    if not cuda_available:
        print("\n[WARNING] CUDA is not available. Training will use CPU.")
        print("  Make sure you installed the cu128 wheel, not the default PyPI one.")
        return

    print(f"[4] CUDA Version    : {torch.version.cuda}")

    # ── 4. GPU details ───────────────────────────────────────
    gpu_name = torch.cuda.get_device_name(0)
    sm_major, sm_minor = torch.cuda.get_device_capability(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print(f"[5] GPU Name        : {gpu_name}")
    print(f"[6] Compute Capab.  : SM {sm_major}.{sm_minor}  <-- Blackwell = 12.0")
    print(f"[7] VRAM Total      : {vram_total:.1f} GB")

    # ── 5. Test tensor on GPU ────────────────────────────────
    # Create a small matrix on GPU, multiply it with itself.
    # If this runs without error, CUDA is fully functional.
    print("\n[8] Running GPU tensor test...")
    x = torch.randn(1000, 1000, device="cuda")   # 1000×1000 random matrix on GPU
    y = x @ x.T                                  # Matrix multiplication
    result_sum = y.sum().item()                   # Bring scalar result back to CPU
    print(f"    Test passed [OK]  (tensor sum: {result_sum:.2f})")

    # ── 6. Memory check ──────────────────────────────────────
    mem_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
    mem_reserved  = torch.cuda.memory_reserved(0)  / (1024 ** 2)
    print(f"[9] GPU Memory Used : {mem_allocated:.1f} MB allocated / {mem_reserved:.1f} MB reserved")

    # ── 7. Summary ───────────────────────────────────────────
    print("\n" + "=" * 55)
    if sm_major == 12:
        print("  [OK] RTX 5080 Blackwell (SM 12.0) confirmed.")
        print("  [OK] PyTorch cu128 build is correct.")
        print("  [OK] This environment is ready for DoomRL training.")
    else:
        print(f"  [WARN] GPU SM is {sm_major}.{sm_minor} - expected 12.0 for RTX 5080.")
    print("=" * 55)


if __name__ == "__main__":
    main()
