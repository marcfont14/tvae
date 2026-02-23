"""
check_gpu.py
============
Verifies that TensorFlow correctly detects the GPU in the NVIDIA Docker container.
Run before any training.

Usage:
    python check_gpu.py
"""

import subprocess
import sys


def check_python_packages():
    print("─" * 50)
    print("  ENVIRONMENT PACKAGES")
    print("─" * 50)
    packages = ['numpy', 'pandas', 'tensorflow', 'pyarrow', 'matplotlib', 'seaborn']
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {pkg:<20} {version}")
        except ImportError:
            print(f"  ✗ {pkg:<20} NOT INSTALLED")


def check_tensorflow_gpu():
    print("\n" + "─" * 50)
    print("  TENSORFLOW + GPU")
    print("─" * 50)

    try:
        import tensorflow as tf

        print(f"  TF version        : {tf.__version__}")
        print(f"  Built with CUDA   : {tf.test.is_built_with_cuda()}")

        gpus = tf.config.list_physical_devices('GPU')
        cpus = tf.config.list_physical_devices('CPU')
        print(f"  CPUs detected     : {len(cpus)}")
        print(f"  GPUs detected     : {len(gpus)}")

        if not gpus:
            print("\n  ✗ No GPU detected.")
            print("    Check that the container was launched with --gpus all")
            return False

        for i, gpu in enumerate(gpus):
            print(f"\n  GPU {i}: {gpu.name}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"    Name           : {details.get('device_name', 'N/A')}")
                print(f"    Compute capability: {details.get('compute_capability', 'N/A')}")
            except Exception:
                pass

        # Enable memory growth to avoid TensorFlow reserving all VRAM
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"\n  ✓ Memory growth enabled on {gpu.name}")
            except RuntimeError as e:
                print(f"  ⚠ Memory growth: {e}")

        return True

    except ImportError:
        print("  ✗ TensorFlow is not installed")
        return False


def check_gpu_compute(n_iterations: int = 100):
    """Runs a real GPU operation and measures runtime."""
    print("\n" + "─" * 50)
    print("  GPU COMPUTE TEST")
    print("─" * 50)

    try:
        import tensorflow as tf
        import time

        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("  ⚠ No GPU, skipping compute test")
            return

        # Reference operation: large matrix multiplication
        size = 4096
        print(f"  Matrix multiplication {size}×{size} ({n_iterations} iterations)")

        # CPU
        with tf.device('/CPU:0'):
            a = tf.random.normal([size, size])
            b = tf.random.normal([size, size])
            _ = tf.matmul(a, b)  # warmup

            t0 = time.perf_counter()
            for _ in range(n_iterations):
                tf.matmul(a, b)
            cpu_time = time.perf_counter() - t0

        # GPU
        with tf.device('/GPU:0'):
            a = tf.random.normal([size, size])
            b = tf.random.normal([size, size])
            _ = tf.matmul(a, b)  # warmup

            t0 = time.perf_counter()
            for _ in range(n_iterations):
                tf.matmul(a, b)
            gpu_time = time.perf_counter() - t0

        print(f"  CPU time   : {cpu_time:.3f}s")
        print(f"  GPU time   : {gpu_time:.3f}s")
        print(f"  Speedup    : {cpu_time / gpu_time:.1f}x")

        if cpu_time / gpu_time > 2:
            print("  ✓ GPU is accelerating correctly")
        else:
            print("  ⚠ Speedup is low, check the configuration")

    except Exception as e:
        print(f"  ✗ Error in compute test: {e}")


def check_nvidia_smi():
    print("\n" + "─" * 50)
    print("  NVIDIA-SMI")
    print("─" * 50)
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=name,memory.total,memory.free,temperature.gpu,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    print(f"  GPU {i}:")
                    print(f"    Name        : {parts[0]}")
                    print(f"    VRAM total  : {parts[1]} MiB")
                    print(f"    VRAM free   : {parts[2]} MiB")
                    print(f"    Temperature : {parts[3]} °C")
                    print(f"    Utilization : {parts[4]} %")
        else:
            print(f"  ✗ nvidia-smi failed: {result.stderr}")
    except FileNotFoundError:
        print("  ✗ nvidia-smi not found")
    except subprocess.TimeoutExpired:
        print("  ✗ nvidia-smi timed out")


def main():
    print("\n" + "=" * 50)
    print("  CHECK GPU — Glucose Foundation Model")
    print("=" * 50)

    check_python_packages()
    gpu_ok = check_tensorflow_gpu()
    check_nvidia_smi()

    if gpu_ok:
        check_gpu_compute()

    print("\n" + "=" * 50)
    if gpu_ok:
        print("  ✓ Environment ready for training")
    else:
        print("  ✗ Review GPU configuration before continuing")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()
