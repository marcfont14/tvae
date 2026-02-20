"""
check_gpu.py
============
Verifica que TensorFlow detecta correctamente la GPU en el docker de NVIDIA.
Ejecutar antes de cualquier entrenamiento.

Uso:
    python check_gpu.py
"""

import subprocess
import sys


def check_python_packages():
    print("─" * 50)
    print("  PAQUETES DEL ENTORNO")
    print("─" * 50)
    packages = ['numpy', 'pandas', 'tensorflow', 'pyarrow', 'matplotlib', 'seaborn']
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {pkg:<20} {version}")
        except ImportError:
            print(f"  ✗ {pkg:<20} NO INSTALADO")


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
        print(f"  CPUs detectadas   : {len(cpus)}")
        print(f"  GPUs detectadas   : {len(gpus)}")

        if not gpus:
            print("\n  ✗ No se detectó ninguna GPU.")
            print("    Comprueba que el contenedor se lanzó con --gpus all")
            return False

        for i, gpu in enumerate(gpus):
            print(f"\n  GPU {i}: {gpu.name}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"    Nombre         : {details.get('device_name', 'N/A')}")
                print(f"    Compute capability: {details.get('compute_capability', 'N/A')}")
            except Exception:
                pass

        # Habilitar memory growth para evitar que TF reserve toda la VRAM
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"\n  ✓ Memory growth activado en {gpu.name}")
            except RuntimeError as e:
                print(f"  ⚠ Memory growth: {e}")

        return True

    except ImportError:
        print("  ✗ TensorFlow no está instalado")
        return False


def check_gpu_compute(n_iterations: int = 100):
    """Hace una operación real en GPU y mide el tiempo."""
    print("\n" + "─" * 50)
    print("  TEST DE CÓMPUTO EN GPU")
    print("─" * 50)

    try:
        import tensorflow as tf
        import time

        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("  ⚠ Sin GPU, saltando test de cómputo")
            return

        # Operación de referencia: multiplicación de matrices grandes
        size = 4096
        print(f"  Multiplicación de matrices {size}×{size} ({n_iterations} iteraciones)")

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

        print(f"  Tiempo CPU : {cpu_time:.3f}s")
        print(f"  Tiempo GPU : {gpu_time:.3f}s")
        print(f"  Speedup    : {cpu_time / gpu_time:.1f}x")

        if cpu_time / gpu_time > 2:
            print("  ✓ GPU está acelerando correctamente")
        else:
            print("  ⚠ El speedup es bajo, revisa la configuración")

    except Exception as e:
        print(f"  ✗ Error en test de cómputo: {e}")


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
                    print(f"    Nombre      : {parts[0]}")
                    print(f"    VRAM total  : {parts[1]} MiB")
                    print(f"    VRAM libre  : {parts[2]} MiB")
                    print(f"    Temperatura : {parts[3]} °C")
                    print(f"    Utilización : {parts[4]} %")
        else:
            print(f"  ✗ nvidia-smi falló: {result.stderr}")
    except FileNotFoundError:
        print("  ✗ nvidia-smi no encontrado")
    except subprocess.TimeoutExpired:
        print("  ✗ nvidia-smi timeout")


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
        print("  ✓ Entorno listo para entrenar")
    else:
        print("  ✗ Revisar configuración GPU antes de continuar")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()
