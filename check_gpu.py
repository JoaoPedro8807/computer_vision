import torch
import sys
import subprocess


print("\n PyTorch:")
print(f"  Versão: {torch.__version__}")

print("\n CUDA:")
cuda_available = torch.cuda.is_available()
print(f"  CUDA Disponível: {cuda_available}")

if cuda_available:
    print(f"  CUDA Version: {torch.version.cuda}")
    
    # 3. Número de GPUs
    gpu_count = torch.cuda.device_count()
    print(f"  Número de GPUs: {gpu_count}")
    
    # 4. Detalhes de cada GPU
    for i in range(gpu_count):
        print(f"\n  GPU {i}:")
        print(f"    Nome: {torch.cuda.get_device_name(i)}")
        print(f"    Capacidade Computacional: {torch.cuda.get_device_capability(i)}")
        print(f"    Memória Total: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    # 5. GPU padrão
    print(f"\n  GPU Padrão: {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})")
    
    # 6. Memória disponível
    print(f"\n  Memória Disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  Memória em Uso: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
else:
    print("GPU não disponível. Usando CPU.")

# 7. Verificar cuDNN
print("\n cuDNN:")
print(f"  Disponível: {torch.backends.cudnn.enabled}")
if cuda_available:
    print(f"  Versão: {torch.backends.cudnn.version()}")

# 8. Teste rápido
print("\n TESTE DE PERFORMANCE:")
print("  Criando tensor de teste...")

try:
    if cuda_available:
        # Teste na GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # Warm-up
        z = torch.mm(x, y)
        
        # Teste
        import time
        start = time.time()
        for _ in range(100):
            z = torch.mm(x, y)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"GPU funcionando! Tempo: {gpu_time:.4f}s")
        
        # Teste na CPU para comparação
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        start = time.time()
        for _ in range(100):
            z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        print(f"  CPU tempo: {cpu_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x mais rápido na GPU")
    else:
        print("Testando apenas CPU")
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        z = torch.mm(x, y)
        print("CPU funcionando normalmente")
        
except Exception as e:
    print(f" Erro: {e}")

# Tentar nvidia-smi
print("\n NVIDIA-SMI:")
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used', '--format=csv,noheader'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("nvidia-smi não encontrado")
except Exception as e:
    print(f"Erro ao executar nvidia-smi: {e}")

print("\n" + "=" * 60)

if cuda_available:
    print("GPU PRONTA PARA USO!")
    print(f"   Use: model = model.cuda() ou device = torch.device('cuda')")
else:
    print("GPU não disponível. Sistema rodará em CPU (mais lento).")
    print("Considere instalar CUDA e cuDNN para melhor performance.")

print("=" * 60)
