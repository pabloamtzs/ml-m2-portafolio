# Módulo 2 Portafolio Implementación
Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework

## Dataset utilizado
Link: https://www.kaggle.com/datasets/andonians/random-linear-regression/data

## Estructura del Portafolio
```bash
ml-m2-portafolio/ 
│ 
├── datasets/
│        └── train.csv 
│        └── test.csv
│
├── regresion-lineal/ 
│         └── portafolio-regresion-lineal.py
│         └── portafolio-regresion-lineal.ipynb 
│
├── venv/
│
```
- regresion-lineal/portafolio-regresion-lineal.py: Código de implementación de la regresión lineal.
- regresion-lineal/portafolio-regresion-lineal.ipynb: Implementación de la regresión lineal en un Jupyter Notebook
- datasets/train.csv: Conjunto de datos de entrenamiento.
- datasets/test.csv: Conjunto de datos de prueba.
- venv/: contiene el entorno virtual con las librerias necesarias.

## Instalación
Para ejecutar el portafolio, primero necesitas clonar este repositorio y luego configurar un entorno virtual para manejar las dependencias.

1. Clonar el repositorio
```bash
git clone <https://github.com/pabloamtzs/ml-m2-portafolio.git>
cd ml-m2-portafolio
```

2. Crear y activar un entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows, usa `venv\Scripts\activate`
```
3. Instalar las dependencias
Todas las dependencias están listadas en el archivo requirements.txt. Para instalarlas, ejecuta:
```bash
pip install -r requirements.txt
```

## Uso
1. Navega a la carpeta regresion-lineal
```bash
cd regresion-lineal
```

2. Ejecuta el script
```bash
python portafolio-regresion-lineal.py
```
