import matplotlib
matplotlib.use('Agg')
from flask import Flask, jsonify
import matplotlib.pyplot as plt
import seaborn as sns
from pykrige.ok import OrdinaryKriging
import numpy as np
import geopandas as gpd
import uuid

def generate_unique_filename(extension=".png"):
    return str(uuid.uuid4()) + extension

app = Flask(__name__)

# Datos de prueba. Reemplaza esto con tus datos o con una solicitud POST para obtener datos del cliente.
data = np.array([[1.0, 1.0, 2.0],
                 [3.0, 1.5, 4.5],
                 [4.5, 5.0, 3.6]])

@app.route('/get_kriging_visualizations', methods=['GET'])
def get_visualizations():
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Semivariograma
    OK = OrdinaryKriging(x, y, z, variogram_model='linear', verbose=False, enable_plotting=True)
    semivariogram_path = "static/" + generate_unique_filename()
    plt.savefig(semivariogram_path)
    plt.clf()

    # Mapa de densidad
    gdf = gpd.GeoDataFrame({'geometry': gpd.points_from_xy(x, y), 'values': z})
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world.plot()
    gdf.plot(ax=ax, markersize=z*100, legend=True)
    density_path = "static/" + generate_unique_filename()
    plt.savefig(density_path)
    plt.clf()

    # En este ejemplo, simplemente devolvemos las rutas locales de las imágenes.
    # En una implementación real, deberías cargar estas imágenes a un servicio de almacenamiento en la nube y devolver esas URLs.
    return jsonify({
        "semivariogram_url": semivariogram_path,
        "density_map_url": density_path
    })

if __name__ == '__main__':
    app.run(debug=True)