#!/bin/bash

# Script para configurar las variables de entorno de CUDA
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "Variables de entorno de CUDA configuradas:"
echo "CUDA_HOME=$CUDA_HOME"
echo "PATH incluye: $CUDA_HOME/bin"
echo "LD_LIBRARY_PATH incluye: $CUDA_HOME/lib64"
echo ""
echo "Para hacer esto permanente, agrega estas líneas a tu ~/.bashrc:"
echo "export CUDA_HOME=/usr/local/cuda-12.6"
echo "export PATH=\$CUDA_HOME/bin:\$PATH"
echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"