# Importar las librerías necesarias
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Especificar la ruta relativa a la carpeta donde se encuentran los datasets
dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')

# Leer los archivos CSV desde la carpeta 'datasets'
df_train = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))
df_test = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))

def update_w_and_b(X, y, w, b, alpha):
    '''Actualiza los parámetros w y b en una época utilizando descenso de gradiente'''
    dl_dw = 0.0  # Gradiente de w
    dl_db = 0.0  # Gradiente de b
    N = len(X)   # Número de observaciones
    for i in range(N):
        y_pred = w * X[i] + b  # Predicción actual
        error = y[i] - y_pred  # Error de la predicción
        dl_dw += -2 * X[i] * error  # Derivada parcial respecto a w
        dl_db += -2 * error         # Derivada parcial respecto a b
    w -= (1 / float(N)) * dl_dw * alpha
    b -= (1 / float(N)) * dl_db * alpha
    return w, b

def avg_loss(X, y, w, b):
    '''Calcula el error cuadrático medio (MSE) para una característica'''
    N = len(X)
    total_error = 0.0
    for i in range(N):
        y_pred = w * X[i] + b
        total_error += (y[i] - y_pred) ** 2
    return total_error / float(N)

def train(X, y, w, b, alpha, epochs):
    '''Itera sobre múltiples épocas e imprime el progreso del entrenamiento'''
    print('Training progress:')
    for e in range(epochs):
        w, b = update_w_and_b(X, y, w, b, alpha)
        if e % 500 == 0:
            avg_loss_ = avg_loss(X, y, w, b)
            print("Epoch {} | Loss: {} | w:{}, b:{}".format(e, avg_loss_, round(w, 4), round(b, 4)))
    return w, b

def train_and_plot(X, y, w, b, alpha, epochs):
    '''Itera sobre múltiples épocas y muestra gráficos que muestran el progreso'''
    for e in range(epochs):
        w, b = update_w_and_b(X, y, w, b, alpha)
        if e == epochs - 1:
            avg_loss_ = avg_loss(X, y, w, b)
            plt.scatter(X, y, color='blue', label='Data')
            y_pred = predict(X, w, b)
            plt.plot(X, y_pred, color='red', label='Model')
            plt.title("Epoch {} | Loss: {} | w:{}, b:{}".format(e, round(avg_loss_, 2), round(w, 4), round(b, 4)))
            plt.xlabel('YearsExperience')
            plt.ylabel('Salary')
            plt.legend()
            plt.show()
    return w, b

def predict(X, w, b):
    '''Realiza predicciones utilizando un modelo lineal simple'''
    return w * X + b

# Bloque principal del script
if __name__ == "__main__":

    # Preprocesar los datos
    df_train.dropna(axis='rows', inplace=True)  # Eliminar filas con valores faltantes
    X = df_train[['x']].values.flatten()
    y = df_train['y'].values
    X_mean = np.mean(X)
    X_std = np.std(X)
    X = (X - X_mean) / X_std
    y_mean = np.mean(y)
    y_std = np.std(y)
    y = (y - y_mean) / y_std

    # Inicializar parámetros y entrenar el modelo
    w = 0
    b = 0
    alpha = 0.001
    epochs = 5000
    w, b = train(X, y, w, b, alpha, epochs)

    # Visualizar el progreso del entrenamiento
    train_and_plot(X, y, w, b, alpha, epochs)

    # Hacer predicciones para un valor específico
    predict_x = 40
    predict_x_scaled = (predict_x - X_mean) / X_std
    predict_y_scaled = predict(predict_x_scaled, w, b)
    predict_y = predict_y_scaled * y_std + y_mean
    print('Para x={}, la predicción de y es y={}'.format(predict_x, round(predict_y, 4)))

    # Probar el modelo con datos de prueba
    X_test = df_test['x'].values.flatten()
    y_test = df_test['y'].values
    X_test_scaled = (X_test - X_mean) / X_std
    y_test_scaled = (y_test - y_mean) / y_std
    y_pred_scaled = predict(X_test_scaled, w, b)
    y_pred = y_pred_scaled * y_std + y_mean
    mse_test = np.mean((y_pred - y_test) ** 2)
    print("Mean Squared Error on test data:", mse_test)

    # Visualizar las predicciones en el conjunto de prueba
    plt.scatter(X_test, y_test, color='blue', label='Test Data')
    plt.plot(X_test, y_pred, color='red', label='Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predictions vs Actual Test Data')
    plt.legend()
    plt.show()
