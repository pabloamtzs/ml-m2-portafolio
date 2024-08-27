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
git clone https://github.com/pabloamtzs/ml-m2-portafolio.git
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

Esto puede tardar un poco la primera vez, pero lo primero que se mostrará es el entrenamiento del modelo con el número de epochs, su respectiva pérdida y el ajuste en las variables 'w' y 'b':

![alt text](<Screenshot 2024-08-26 at 23.07.49.png>) 

Una vez finalizado el entrenamiento, se mostrará la primera gráfica, la cual representa nuestro modelo en el último epoch, mostrando los valores óptimos para el dataset de entrenamiento:
![alt text](<Screenshot 2024-08-26 at 23.08.26.png>) 

Al cerrar la gráfica anterior, se desplegará la siguiente gráfica, que mostrará nuestro modelo (nuestra predicción) junto a los datos de prueba:
![alt text](<Screenshot 2024-08-26 at 23.08.34.png>)

Finalmente, podemos ver una predicción de nuestro modelo en un punto específico de la gráfica, además de mostrarnos el valor de nuestro MSE en el dataset de prueba. Un MSE de 9.43 es relativamente bajo, lo que implica que, en promedio, la diferencia al cuadrado entre las predicciones y los valores reales no es significativa. Esto indica que el modelo predice con bastante precisión los valores en el conjunto de datos de prueba, lo cual es una señal de buen rendimiento.

![alt text](<Screenshot 2024-08-26 at 23.08.03.png>) 

Dentro de la carpeta de regresion-lineal, en el archivo portafolio-regresion-lineal.py, en la línea 92, puedes modificar el valor de x para probar otras predicciones.
