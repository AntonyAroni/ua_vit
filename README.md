# Vision Transformer (ViT) for Fashion-MNIST

Un Vision Transformer implementado desde cero en C++ con CUDA para clasificación de imágenes Fashion-MNIST.

## 📋 Descripción

Este proyecto implementa un Vision Transformer (ViT) completamente funcional usando C++ y CUDA para acelerar el entrenamiento en GPU. El modelo está diseñado para clasificar imágenes del dataset Fashion-MNIST (10 clases de ropa).

### Características principales:
- **Implementación desde cero**: Vision Transformer completo sin dependencias de frameworks de ML
- **Aceleración GPU**: Operaciones matriciales optimizadas con cuBLAS
- **Arquitectura modular**: Código organizado en clases reutilizables
- **Entrenamiento completo**: Forward pass, backward pass y optimización Adam
- **Métricas en tiempo real**: Loss y accuracy durante el entrenamiento

## 🏗️ Arquitectura del Modelo

- **Patch Size**: 4x4 píxeles
- **Embedding Dimension**: 128
- **Transformer Depth**: 2 capas
- **Attention Heads**: 4
- **MLP Hidden Dimension**: 256 (2x embedding dim)
- **Batch Size**: 64
- **Learning Rate**: 3e-3 con cosine annealing

## 🛠️ Requisitos del Sistema

### Hardware
- GPU NVIDIA con compute capability 3.5+
- Mínimo 4GB de VRAM
- 8GB+ de RAM del sistema

### Software
- **CUDA Toolkit 11.0+** (probado con 12.6)
- **CMake 3.18+**
- **GCC 9.0+** o compilador compatible con C++17
- **cuBLAS** (incluido con CUDA Toolkit)

### Verificar instalación CUDA:
```bash
nvcc --version
nvidia-smi
```

## 📦 Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd ia_vit
```

### 2. Configurar CUDA (si es necesario)
```bash
chmod +x setup_cuda.sh
./setup_cuda.sh
```

### 3. Descargar el dataset Fashion-MNIST
```bash
mkdir -p data
cd data

# Descargar archivos
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

# Descomprimir
gunzip *.gz
cd ..
```

### 4. Compilar el proyecto
```bash
mkdir -p build
cd build

# Configurar con CMake
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75

# Compilar
make -j$(nproc)
```

**Nota**: Ajusta `CMAKE_CUDA_ARCHITECTURES` según tu GPU:
- GTX 1650/1660: `75`
- RTX 20xx: `75`
- RTX 30xx: `86`
- RTX 40xx: `89`

## 🚀 Ejecución

### Entrenar el modelo
```bash
cd build
./ia_vit
```

### Salida esperada
```
=== Vision Transformer for Fashion-MNIST ===
Initializing CUDA...
Using GPU: NVIDIA GeForce GTX 1650
Memory: 4096 MB
Compute Capability: 7.5

=== Model Configuration ===
Batch size: 64
Epochs: 50
Learning rate: 0.003
Image size: 28x28
Patch size: 4x4
Embedding dimension: 128
Transformer depth: 2
Number of heads: 4

=== Loading Fashion-MNIST Dataset ===
Training samples: 60000
Test samples: 10000

=== Starting Training ===
Epoch   1/50 | Time: 2.6s
Train Loss: 2.1378 | Train Acc: 28.31%
Val Acc: 37.40%
Learning Rate: 2.87e-03
```

## 📊 Rendimiento Esperado

### Métricas de entrenamiento típicas:
- **Época 1**: Loss ~2.13, Accuracy ~28%
- **Época 10**: Loss ~2.05, Accuracy ~35%
- **Época 20**: Loss ~1.85, Accuracy ~42%
- **Época 50**: Loss ~1.45, Accuracy ~48%

### Tiempo de entrenamiento:
- **GTX 1650**: ~2.6s por época
- **RTX 3070**: ~1.2s por época
- **RTX 4090**: ~0.8s por época

## 🗂️ Estructura del Proyecto

```
ia_vit/
├── main.cpp                 # Punto de entrada y loop de entrenamiento
├── vit_transformer.h/cpp    # Implementación del Vision Transformer
├── data_loader.h/cpp        # Cargador del dataset Fashion-MNIST
├── cuda_kernels.cu          # Kernels CUDA personalizados
├── regularization.h         # Técnicas de regularización
├── CMakeLists.txt          # Configuración de compilación
├── setup_cuda.sh          # Script de configuración CUDA
├── data/                   # Dataset Fashion-MNIST
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── build/                  # Archivos compilados
    └── ia_vit             # Ejecutable
```

## 🔧 Personalización

### Modificar hiperparámetros
Edita las constantes en `main.cpp`:
```cpp
const int batch_size = 64;
const int epochs = 50;
const float initial_lr = 3e-3f;
const int embed_dim = 128;
const int depth = 2;
const int num_heads = 4;
```

### Cambiar arquitectura
Modifica la clase `VisionTransformer` en `vit_transformer.h/cpp` para:
- Agregar más capas transformer
- Cambiar dimensiones de embedding
- Implementar diferentes tipos de atención

## 🐛 Solución de Problemas

### Error: "No CUDA devices found"
```bash
# Verificar drivers NVIDIA
nvidia-smi

# Reinstalar CUDA Toolkit
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

### Error: "CMAKE_CUDA_COMPILER could not be found"
```bash
# Encontrar nvcc
which nvcc
# Usar la ruta completa en cmake
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Error: "Failed to load data"
```bash
# Verificar que el dataset esté en la ubicación correcta
ls -la data/
# Debe mostrar los 4 archivos .ubyte
```

### Memoria GPU insuficiente
- Reducir `batch_size` en `main.cpp`
- Reducir `embed_dim` o `depth`

## 📈 Mejoras Futuras

- [ ] Implementar data augmentation
- [ ] Agregar soporte para múltiples GPUs
- [ ] Optimizar kernels CUDA personalizados
- [ ] Implementar diferentes schedulers de learning rate
- [ ] Agregar técnicas de regularización avanzadas
- [ ] Soporte para otros datasets (CIFAR-10, ImageNet)

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📞 Soporte

Si encuentras problemas o tienes preguntas:
- Abre un issue en GitHub
- Revisa la sección de solución de problemas
- Verifica que tu sistema cumple los requisitos

---

**Desarrollado usando C++ y CUDA**