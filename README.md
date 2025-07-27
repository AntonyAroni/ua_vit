# Vision Transformer (ViT) para Fashion-MNIST en C++/CUDA

Este proyecto implementa un Vision Transformer completamente optimizado en C++ con kernels CUDA para entrenar en el dataset Fashion-MNIST.

## Características

- **Implementación completa en C++/CUDA**: Todo el modelo está implementado desde cero
- **Kernels CUDA optimizados**: Operaciones matriciales y de atención paralelizadas
- **Patch Embedding**: Conversión eficiente de imágenes a patches con convolución
- **Multi-Head Attention**: Implementación paralela del mecanismo de atención
- **Forward y Backward Pass**: Propagación hacia adelante y hacia atrás completas
- **Data Augmentation**: Rotación, flip horizontal y ruido
- **Optimizaciones GPU**: Uso de cuBLAS, cuDNN y mixed precision
- **Scheduler de learning rate**: Cosine annealing con warmup

## Requisitos

- CUDA Toolkit 11.0+
- cuBLAS
- cuDNN
- CMake 3.18+
- GPU con compute capability 7.0+ (recomendado)
- Compilador C++17 compatible

## Compilación

```bash
# Crear directorio de build
mkdir build && cd build

# Configurar con CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compilar
make -j$(nproc)

# Descargar dataset (opcional)
make download_data
```

## Estructura del Proyecto

```
ia_vit/
├── main.cpp              # Loop principal de entrenamiento
├── vit_transformer.h     # Declaraciones del modelo ViT
├── vit_transformer.cpp   # Implementación del modelo
├── cuda_kernels.cu       # Kernels CUDA optimizados
├── data_loader.h         # Cargador de datos
├── data_loader.cpp       # Implementación del data loader
├── CMakeLists.txt        # Configuración de compilación
└── README.md            # Este archivo
```

## Configuración del Modelo

- **Tamaño de imagen**: 28x28 (Fashion-MNIST)
- **Tamaño de patch**: 4x4
- **Dimensión de embedding**: 192
- **Profundidad**: 12 capas transformer
- **Número de heads**: 8
- **Clases**: 10 (categorías Fashion-MNIST)

## Uso

1. **Preparar datos**: Coloca los archivos de Fashion-MNIST en `./data/`:
   - `train-images-idx3-ubyte`
   - `train-labels-idx1-ubyte`
   - `t10k-images-idx3-ubyte`
   - `t10k-labels-idx1-ubyte`

2. **Ejecutar entrenamiento**:
   ```bash
   ./ia_vit
   ```

## Dataset Fashion-MNIST

Fashion-MNIST es un dataset de 70,000 imágenes en escala de grises de 28x28 píxeles:
- 60,000 imágenes de entrenamiento
- 10,000 imágenes de prueba
- 10 clases: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

Puedes descargar el dataset desde:
https://github.com/zalandoresearch/fashion-mnist

## Optimizaciones CUDA

### Kernels Implementados

1. **Patch Embedding**: Extracción paralela de patches con convolución
2. **Multi-Head Attention**: Cálculo paralelo de Q, K, V y atención escalada
3. **Layer Normalization**: Normalización por capas optimizada
4. **GELU Activation**: Función de activación GELU con aproximación rápida
5. **Cross-Entropy Loss**: Cálculo de pérdida con softmax estable

### Optimizaciones de Memoria

- Uso de memoria unificada cuando es posible
- Reutilización de buffers temporales
- Transferencias asíncronas CPU-GPU
- Compilación con separación de dispositivo

### Optimizaciones de Cómputo

- Fast math habilitado (`--use_fast_math`)
- Múltiples arquitecturas de GPU soportadas
- Uso de streams CUDA para paralelización
- Optimizaciones específicas por arquitectura

## Parámetros de Entrenamiento

- **Batch size**: 128
- **Learning rate inicial**: 3e-4
- **Épocas**: 100
- **Optimizer**: AdamW
- **Scheduler**: Cosine annealing con 10% warmup
- **Data augmentation**: Flip horizontal, rotación ±10°, ruido gaussiano

## Rendimiento Esperado

En una GPU moderna (RTX 3080/4080), deberías obtener:
- **Precisión de entrenamiento**: ~95%+
- **Precisión de validación**: ~92%+
- **Tiempo por época**: ~30-60 segundos
- **Convergencia**: ~20-50 épocas

## Arquitectura del Modelo

```
Input (28x28x1) 
    ↓
Patch Embedding (49 patches de 4x4 → 192D)
    ↓ 
Add Positional Embedding + Class Token
    ↓
12x Transformer Blocks:
  ├── Layer Norm
  ├── Multi-Head Attention (8 heads)
  ├── Residual Connection
  ├── Layer Norm  
  ├── MLP (192 → 768 → 192)
  └── Residual Connection
    ↓
Layer Norm
    ↓
Classification Head (192 → 10)
    ↓
Softmax → Class Prediction
```

## Monitoreo durante Entrenamiento

El programa muestra en tiempo real:
- Progreso de la época con barra visual
- Loss y accuracy por batch
- Tiempo por batch/época
- Learning rate actual
- Accuracy de validación por época

## Solución de Problemas

### Error: No CUDA devices found
- Verifica que los drivers de NVIDIA estén instalados
- Ejecuta `nvidia-smi` para verificar la GPU

### Error: cuBLAS/cuDNN not found
- Instala CUDA Toolkit completo
- Verifica las variables de entorno CUDA_HOME

### Out of memory
- Reduce el batch size en `main.cpp`
- Reduce la profundidad del modelo o embed_dim

### Compilación lenta
- Usa `make -j$(nproc)` para compilación paralela
- Considera reducir las arquitecturas CUDA objetivo

## Extensiones Posibles

- Implementar mixed precision (FP16)
- Agregar más técnicas de data augmentation
- Implementar attention visualization
- Soporte para datasets más grandes
- Implementar model checkpointing
- Agregar métricas adicionales (F1, precision, recall)

## Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.