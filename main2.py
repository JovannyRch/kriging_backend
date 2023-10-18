from flask import Flask, jsonify
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import numpy as np

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
    semivariogram_path = "static/semivariogram.png"
    plt.savefig(semivariogram_path)
    plt.clf()

    # Mapa de densidad
    gdf = gpd.GeoDataFrame({'geometry': gpd.points_from_xy(x, y), 'values': z})
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world.plot()
    gdf.plot(ax=ax, markersize=z*100, legend=True)
    density_path = "static/density_map.png"
    plt.savefig(density_path)
    plt.clf()


    return jsonify({
        "semivariogram_url": semivariogram_path,
        "density_map_url": density_path
    })

if __name__ == '__main__':
    app.run(debug=True)