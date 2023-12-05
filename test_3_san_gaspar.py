import numpy as np
import matplotlib.pyplot as plt
from skgstat import Variogram, OrdinaryKriging
plt.style.use('ggplot')


# Leer los datos del archivo
data = np.loadtxt('3_san_gaspar.txt')

# Separar las coordenadas y los valores
coordinates = data[:, 0:2]
values = data[:, 2]

# Crear el objeto Variogram
V = Variogram(coordinates, values, maxlag='median',
              n_lags=5, normalize=False, verbose=True)
fig = V.plot(show=False)


plt.show()
