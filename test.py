import numpy as np
import matplotlib.pyplot as plt
from skgstat import Variogram, OrdinaryKriging
import pandas as pd


def plot_scatter(coordinates, values, ax):
    x = coordinates[:, 0]
    y = coordinates[:, 1]

    art = ax.scatter(x, y, 50, c=values, cmap='plasma')
    plt.colorbar(art, ax=ax)


def interpolate(V, ax):
    xx, yy = np.mgrid[0:499:100j, 0:499:100j]
    ok = OrdinaryKriging(V, min_points=5, max_points=15, mode='exact')
    field = ok.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
    art = ax.matshow(field, origin='lower', cmap='plasma',
                     vmin=V.values.min(), vmax=V.values.max())
    ax.set_title('%s model' % V.model.__name__)
    plt.colorbar(art, ax=ax)
    return field


# Leer los datos del archivo
data = np.loadtxt('queretaro.txt')

# Separar las coordenadas y los valores
coordinates = data[:, 0:2]
values = data[:, 2]

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
plot_scatter(coordinates, values, axes[0])


# Crear el objeto Variogram
V = Variogram(coordinates, values, maxlag='median',
              n_lags=5, normalize=False, verbose=True, model='spherical')
V.plot(show=False)

# Print model variogram values
print(V)
print(V.parameters)
[range, sill, nugget] = V.parameters

print("range: ", range)
print("sill: ", sill)
print("nugget: ", nugget)


fields = []
fig, _a = plt.subplots(1, 2, figsize=(12, 10), sharex=True, sharey=True)
fields.append(interpolate(V, _a[0]))

pd.DataFrame({'spherical': fields[0].flatten()})


# Calculamos manualmente los bines y valores de semivarianza
bins = V.bins
experimental_semivariance = V.experimental
# insert 0 at the beginning of the array
bins = np.insert(bins, 0, 0)
experimental_semivariance = np.insert(experimental_semivariance, 0, 0)


# Graficar el semivariograma experimental
""" plt.figure(figsize=(6, 4))
plt.plot(bins, experimental_semivariance, 'o-')
plt.title('Semivariograma Experimental')
plt.xlabel('Distancia')
plt.ylabel('Semivarianza')
plt.grid(True) """
plt.show()


""" 

    Incio Ajuste -> Semivariograma Experimental
    Ajuste de modelo -> Semivariograma Teórico

 """
