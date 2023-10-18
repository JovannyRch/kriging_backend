from pykrige.ok import OrdinaryKriging
import numpy as np

# Datos de muestra
lons = np.array([1.0, 2.0, 3.0, 4.0])
lats = np.array([2.0, 3.0, 4.0, 5.0])
values = np.array([10, 20, 15, 25])

# Realizar kriging ordinario
OK = OrdinaryKriging(lons, lats, values, variogram_model='spherical')

# Obtener el semivariograma experimental
semivariogram = OK.lags, OK.semivariance

print(semivariogram)