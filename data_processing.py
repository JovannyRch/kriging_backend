import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from skgstat import Variogram, OrdinaryKriging


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
        'nugget': nugget,
        'model': model
    }

# coordinates, values, variogram_model, n_lags, range_val, sill, nugget


def get_custom_semivariogram(coordinates, values, model='spherical', n_lags=10,
                             range_val=None, sill=None, nugget=None, manual_fit=False, return_obj=False):
    # Configurar el método de ajuste y los parámetros
    fit_method = 'manual' if manual_fit else None

    # Crear el objeto Variogram
    V = Variogram(coordinates, values, maxlag='median', n_lags=n_lags,
                  normalize=False, verbose=True, model=model,
                  fit_method=fit_method,
                  fit_range=range_val, fit_sill=sill, fit_nugget=nugget)

    # Ejecutar el ajuste manual si es necesario
    if manual_fit:
        V.fit()

    if return_obj:
        return V

    V.plot(show=False)
    image_base64 = get_image(plt)

    # Obtener los parámetros del semivariograma, ya sea ajustados automáticamente o manualmente
    [range_val, sill, nugget] = V.parameters

    return {
        'lags': V.bins.tolist(),
        'semivariance': V.experimental.tolist(),
        'image_base64': image_base64,
        'range': range_val,
        'sill': sill,
        'nugget': nugget,
        'model': model
    }


def interpolate(V, xx, yy):

    ok = OrdinaryKriging(V, min_points=5, max_points=15, mode='exact')

    # return base 64 image
    field = ok.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)
    fig, ax = plt.subplots()
    art = ax.matshow(field, origin='lower', cmap='jet',
                     vmin=V.values.min(), vmax=V.values.max())

    plt.colorbar(art, ax=ax)
    image_base64 = get_image(plt)
    return image_base64


def get_contour_map(coordinates, values, model='spherical', n_lags=10,
                    range_val=None, sill=None, nugget=None):

    xs = coordinates[:, 0]
    ys = coordinates[:, 1]

    xx, yy = np.mgrid[np.min(xs):np.max(xs):100j,
                      np.min(ys):np.max(ys):100j]

    V = get_custom_semivariogram(
        coordinates, values, model, n_lags, range_val, sill, nugget, True, True)

    # print parameters
    print(V.parameters)

    image_base64 = interpolate(V, xx, yy)

    return {
        'image_base64': image_base64
    }
