import matplotlib
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import numpy as np

# Datos de muestra
lons = np.array([1.0, 2.0, 3.0, 4.0])
lats = np.array([2.0, 3.0, 4.0, 5.0])
values = np.array([10, 20, 15, 25])


x = lons
y = lats
cu = values

# Realizar kriging ordinario
OK = OrdinaryKriging(lons, lats, values, variogram_model='spherical', nlags=20, 
                     variogram_parameters= {'sill': 0.22, 'range': 175, 'nugget': 0}, enable_plotting=True, 
                    coordinates_type='euclidean')

# Obtener el semivariograma experimental
semivariogram = OK.lags, OK.semivariance



#ploteo:

marker_size = 15
plt.scatter(x, y, marker_size, cu, cmap=plt.cm.Blues) #el estilo gist_rainbow es algo mas convencional.
plt.xlabel("Este [X]")
plt.ylabel("Norte [Y]")
plt.title("Visualización 2D NE muestras Cu [%]")
cbar = plt.colorbar()
cbar.set_label("Cu [%]", labelpad=+1)
plt.show()

print(semivariogram)