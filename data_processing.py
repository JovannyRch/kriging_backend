import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from skgstat import Variogram


def get_image(plt, format='png'):
    # Guarda la figura en un buffer en lugar de en un archivo
    buf = BytesIO()
    plt.savefig(buf, format=format)
    plt.close()
    buf.seek(0)

    # Codifica la imagen para enviar en base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return image_base64


def plot_scatter_to_base64(coordinates, values):
    # Crear la figura y el eje para el gráfico
    fig, ax = plt.subplots()

    # Crear el gráfico de dispersión
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    scatter = ax.scatter(x, y, 50, c=values, cmap='plasma')
    plt.colorbar(scatter, ax=ax)

    image_base64 = get_image(plt)

    return image_base64


def get_semivariogram(coordinates, values, model='spherical', n_lags=10):
    V = Variogram(coordinates, values, maxlag='median',
                  n_lags=n_lags, normalize=False, verbose=True, model=model)

    V.plot(show=False)

    image_base64 = get_image(plt)

    [range_val, sill, nugget] = V.parameters

    return {
        'lags': V.bins.tolist(),
        'semivariance': V.experimental.tolist(),
        'image_base64': image_base64,
        'range': range_val,
        'sill': sill,
        'nugget': nugget
    }
