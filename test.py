import numpy as np
import matplotlib.pyplot as plt
from skgstat import Variogram


# Leer los datos del archivo
data = np.loadtxt('points2.txt')

# Separar las coordenadas y los valores
coordinates = data[:, 0:2]
values = data[:, 2]

# Crear el objeto Variogram
V = Variogram(coordinates, values, model='spherical',
              fit_method=None, n_lags=13)

# Calculamos manualmente los bines y valores de semivarianza
bins = V.bins
experimental_semivariance = V.experimental

# Graficar el semivariograma experimental
plt.figure(figsize=(6, 4))
plt.plot(bins, experimental_semivariance, 'o-')
plt.title('Semivariograma Experimental')
plt.xlabel('Distancia')
plt.ylabel('Semivarianza')
plt.grid(True)
plt.show()
