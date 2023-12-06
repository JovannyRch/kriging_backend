import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO


def plot_scatter_to_base64(coordinates, values):
    # Crear la figura y el eje para el gráfico
    fig, ax = plt.subplots()

    # Crear el gráfico de dispersión
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    scatter = ax.scatter(x, y, 50, c=values, cmap='plasma')
    plt.colorbar(scatter, ax=ax)

    # Guardar el gráfico en un buffer en formato PNG
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)  # Cerrar la figura para liberar memoria
    buf.seek(0)

    # Convertir la imagen en el buffer a una cadena base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return image_base64
