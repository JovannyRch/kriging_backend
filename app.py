
from skgstat import Variogram
from flask_cors import CORS
from io import BytesIO
import base64
import uuid
import os
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from flask import Flask, jsonify, request
import matplotlib

import const
import data_processing

matplotlib.use('Agg')
plt.style.use('ggplot')


app = Flask(__name__)
CORS(app)

testing_points = const.POINTS


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/semivariogram', methods=['POST'])
def get_semivariogram():
    # Recibe los datos en formato JSON desde Flutter
    data = request.get_json()
    variogram_model = data.get('variogram_model', 'spherical')
    n_lags = data.get('n_lags', 13)

    points = np.array(data["points"])

    if data.get('testing', False):
        points = np.loadtxt('testing_points.txt')

    # Separar las coordenadas y los valores
    coordinates = points[:, 0:2]
    values = points[:, 2]

    response = data_processing.get_semivariogram(
        coordinates, values, variogram_model, n_lags)

    return jsonify(response)

# Add custom semivariogram


@app.route('/custom_semivariogram', methods=['POST'])
def get_custom_semivariogram():
    # Recibe los datos en formato JSON desde Flutter
    data = request.get_json()
    variogram_model = data.get('variogram_model', 'spherical')
    n_lags = data.get('n_lags', 13)
    sill = data.get('sill', 1)
    nugget = data.get('nugget', 0)
    range_val = data.get('range', 1)

    points = np.array(data["points"])

    if data.get('testing', False):
        points = np.loadtxt('testing_points.txt')

    # Separar las coordenadas y los valores
    coordinates = points[:, 0:2]
    values = points[:, 2]

    response = data_processing.get_custom_semivariogram(
        coordinates, values, variogram_model, n_lags, range_val, sill, nugget, True)

    return jsonify(response)


@app.route('/scatter', methods=['POST'])
def get_scatter():
    data = request.get_json()
    points = np.array(data["points"])

    if data.get('testing', False):
        points = np.loadtxt('testing_points.txt')

    coordinates = points[:, 0:2]
    values = points[:, 2]

    image_base64 = data_processing.plot_scatter_to_base64(coordinates, values)

    return jsonify({'image_base64': image_base64})


@app.route('/generate_contour', methods=['POST'])
def generate_contour():
    data = request.get_json()

    points = np.array(data["points"])
    variogram_model = data.get('variogram_model', 'linear')
    model_params = data.get('model_params', None)
    grid_space = data.get('grid_space')
    is_utm = data.get('coordinates', 'latlng') == 'utm'

    if data.get('testing', False):
        points = np.loadtxt('testing_points.txt')

    lons = points[:, 0]  # if is_utm this is easting
    lats = points[:, 1]  # if is_utm this is northing
    values = points[:, 2]

    if model_params and variogram_model == 'linear':
        model_params['slope'] = (
            model_params['sill'] - model_params['nugget']) / model_params['range']
        del model_params['sill']
        del model_params['range']

    if not grid_space:
        lon_range = np.max(lons) - np.min(lons)
        lat_range = np.max(lats) - np.min(lats)
        grid_space = max(lon_range, lat_range) * 0.01

    # Crea una grilla regular sobre el dominio de los datos
    grid_lon = np.arange(np.min(lons), np.max(lons), grid_space)
    grid_lat = np.arange(np.min(lats), np.max(lats), grid_space)

    # Crea una instancia de OrdinaryKriging con los parámetros del modelo seleccionados
    OK = OrdinaryKriging(
        lons, lats, values,
        variogram_model=variogram_model,
        variogram_parameters=model_params
    )

    # Realiza la kriging sobre la grilla definida
    z, ss = OK.execute('grid', grid_lon, grid_lat)

    # Genera el mapa de contornos usando los resultados de kriging
    X, Y = np.meshgrid(grid_lon, grid_lat)
    Z = z.reshape(X.shape)

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, cmap='jet', levels=50)
    plt.colorbar(contour)
    plt.title('Mapa de Contornos de Kriging')
    xLabel = is_utm and 'Este' or 'Longitud'
    yLabel = is_utm and 'Norte' or 'Latitud'
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    # Marca los puntos de datos originales
    plt.scatter(lons, lats, c='red', marker='o')

    # Guarda la figura en un buffer en lugar de en un archivo
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Codifica la imagen para enviar en base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return jsonify({'image_base64': image_base64})


@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap():
    data = request.get_json()

    variogram_model = data.get('variogram_model', 'spherical')
    n_lags = data.get('n_lags', 13)
    sill = data.get('sill', 1)
    nugget = data.get('nugget', 0)
    range_val = data.get('range', 1)

    points = np.array(data["points"])

    if data.get('testing', False):
        points = np.loadtxt('testing_points.txt')

    values = points[:, 2]
    coordinates = points[:, 0:2]

    response = data_processing.get_contour_map(
        coordinates, values, variogram_model, n_lags, range_val, sill, nugget)

    return jsonify(response)


@app.route('/points', methods=['GET'])
def getPoints():
    testing_points = np.loadtxt('testing_points.txt')
    return jsonify({"points": testing_points})


if __name__ == '__main__':
    app.run(debug=True)
