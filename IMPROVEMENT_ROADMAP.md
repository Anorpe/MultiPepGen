# Roadmap de Mejora para Publicación Q1 - MultiPepGen

Este documento detalla las tareas necesarias para elevar la calidad del repositorio al estándar de una publicación en revistas de alto impacto (Q1).

## 🚀 Fase 1: Reproducibilidad y Despliegue
- [x] **Dockerización**: Crear `Dockerfile` y `docker-compose.yml` para garantizar que el entorno de ejecución sea idéntico en cualquier máquina.
- [x] **Gestión de Pesos**: Definir un protocolo para cargar pesos pre-entrenados del modelo final presentado en el artículo.

## 🛠 Fase 2: Interfaz de Usuario y Usabilidad
- [x] **Implementación de CLI**: Crear `src/multipepgen/cli.py` para permitir el entrenamiento y generación desde la terminal (ej: `multipepgen train --config ...`).
- [x] **Notebook de Tutorial**: Crear un Jupyter Notebook en `notebooks/reproducibilidad_figuras.ipynb` que replique una o dos figuras clave del artículo usando el modelo pre-entrenado.

## 🏗 Fase 3: Robustez del Código y Refactorización
- [x] **Migración a Logging**: Reemplazar los `print()` por el módulo `logging` de Python para un control profesional de la salida por consola.
- [x] **Eliminación de Hardcoding**: Asegurar que todos los hiperparámetros (como `max_len`) se lean estrictamente desde los archivos de configuración en `configs/`.
- [x] **Documentación de API**: Agregar Docstrings completos en formato Google o NumPy a todos los métodos y clases del modelo cGAN.

## 📊 Fase 4: Validación y Benchmarking
- [x] **Scripts de Benchmark**: Automatizar la comparación contra otros métodos generativos (ej. Simple GAN, VAE) para facilitar la revisión por pares.
- [x] **Exportación de Métricas**: Implementar la exportación de resultados de validación en formatos estándar (JSON/CSV) para análisis externo.
