
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
matplotlib.use('Agg')
plt.style.use('ggplot')


def generate_unique_filename(extension=".png"):
    return str(uuid.uuid4()) + extension


app = Flask(__name__)
CORS(app)

testing_points = [
    [417297, 2092259, 40],
    [417282, 2092255, 10],
    [417277, 2092253, 0],
    [417265, 2092256, 12],
    [417249, 2092256, 4],
    [417244, 2092257, 20],
    [417221, 2092254, 0],
    [417216, 2092255, 0],
    [417216, 2092256, 0],
    [417204, 2092254, 10],
    [417204, 2092263, 12],
    [417208, 2092262, 8],
    [417224, 2092264, 3],
    [417229, 2092263, 4],
    [417246, 2092265, 28],
    [417245, 2092266, 33],
    [417255, 2092264, 27],
    [417264, 2092260, 0],
    [417298, 2092274, 0],
    [417286, 2092264, 2],
    [417281, 2092269, 0],
    [417267, 2092269, 11],
    [417262, 2092274, 10],
    [417249, 2092273, 0],
    [417236, 2092270, 5],
    [417228, 2092271, 0],
    [417214, 2092269, 7],
    [417206, 2092268, 15],
    [417211, 2092279, 0],
    [417216, 2092279, 6],
    [417297, 2092279, 0],
    [417303, 2092285, 0],
    [417278, 2092285, 0],
    [417273, 2092283, 0],
    [417261, 2092285, 0],
    [417253, 2092287, 0],
    [417227, 2092290, 0],
    [417218, 2092287, 0],
    [417218, 2092289, 15],
    [417216, 2092289, 0],
    [417206, 2092301, 3],
    [417216, 2092298, 3],
    [417232, 2092301, 0],
    [417235, 2092299, 0],
    [417253, 2092299, 0],
    [417260, 2092294, 0],
    [417273, 2092299, 0],
    [417283, 2092296, 0],
    [417295, 2092298, 0],
    [417298, 2092297, 0],
    [417229, 2092309, 0],
    [417222, 2092308, 0],
    [417219, 2092325, 0],
    [417215, 2092316, 0],
    [417243, 2092320, 0],
    [417247, 2092316, 0],
    [417257, 2092316, 0],
    [417268, 2092315, 0],
    [417275, 2092322, 0],
    [417266, 2092324, 0],
    [417258, 2092322, 0],
    [417256, 2092325, 0],
    [417240, 2092323, 0],
    [417234, 2092323, 0],
    [417244, 2092334, 0],
    [417248, 2092334, 0],
    [417259, 2092334, 20],
    [417267, 2092333, 5],
    [417276, 2092332, 0],
    [417284, 2092335, 0],
    [417289, 2092334, 0],
    [417296, 2092333, 0],
    [417306, 2092342, 0],
    [417299, 2092345, 0],
    [417289, 2092347, 0],
    [417279, 2092346, 0],
    [417259, 2092343, 0],
    [417249, 2092347, 0],
    [417222, 2092357, 12],
    [417220, 2092355, 0],
    [417222, 2092361, 17],
    [417229, 2092354, 22],
    [417256, 2092354, 36],
    [417261, 2092356, 0],
    [417276, 2092354, 8],
    [417285, 2092355, 0],
    [417292, 2092354, 0],
    [417301, 2092349, 0],
    [417360, 2092358, 12],
    [417362, 2092363, 12],
    [417360, 2092363, 20],
    [417353, 2092365, 30],
    [417346, 2092367, 32],
    [417336, 2092366, 36],
    [417332, 2092364, 39],
    [417321, 2092369, 29],
    [417315, 2092374, 100],
    [417307, 2092376, 50],
    [417303, 2092375, 50],
    [417296, 2092378, 64],
    [417292, 2092378, 65],
    [417285, 2092379, 80],
    [417285, 2092376, 90],
    [417270, 2092375, 75],
    [417268, 2092372, 60],
    [417257, 2092372, 80],
    [417254, 2092370, 88],
    [417247, 2092369, 50],
    [417233, 2092368, 70],
    [417227, 2092368, 70],
    [417223, 2092381, 45],
    [417220, 2092382, 64],
    [417214, 2092381, 62],
    [417204, 2092383, 85],
    [417201, 2092381, 70],
    [417188, 2092383, 62],
    [417184, 2092380, 65],
    [417183, 2092383, 130],
    [417174, 2092384, 130],
    [417176, 2092381, 130],
    [417193, 2092391, 45],
    [417211, 2092394, 77],
    [417214, 2092395, 0],
    [417237, 2092393, 36],
    [417241, 2092393, 36],
    [417244, 2092389, 42],
    [417247, 2092386, 40],
    [417281, 2092394, 23],
    [417283, 2092393, 0],
    [417291, 2092390, 0],
    [417293, 2092388, 0],
    [417302, 2092385, 50],
    [417304, 2092384, 15],
    [417315, 2092384, 27],
    [417312, 2092392, 0],
    [417308, 2092394, 7],
    [417303, 2092396, 32],
    [417300, 2092395, 19],
    [417288, 2092393, 7],
    [417287, 2092390, 23],
    [417275, 2092390, 18],
    [417260, 2092394, 33],
    [417256, 2092392, 8],
    [417245, 2092390, 7],
    [417237, 2092395, 12],
    [417232, 2092402, 20],
    [417213, 2092403, 44],
    [417206, 2092402, 17],
    [417205, 2092392, 20],
    [417185, 2092399, 0],
    [417196, 2092407, 18],
    [417218, 2092408, 14],
    [417236, 2092411, 0],
    [417238, 2092415, 0],
    [417262, 2092416, 17],
    [417263, 2092411, 24],
    [417273, 2092413, 22],
    [417278, 2092410, 14],
    [417297, 2092407, 22],
    [417293, 2092419, 0]
]


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/semivariogram', methods=['POST'])
def get_semivariogram():
    # Recibe los datos en formato JSON desde Flutter
    data = request.get_json()
    variogram_model = data.get('variogram_model', 'linear')
    # Add testing flag, if true, then use test data calling getPoints() function

    points = np.array(data["points"])

    if data.get('testing', False):
        points = np.array(testing_points)

    lons = points[:, 0]
    lats = points[:, 1]
    values = points[:, 2]

    # Define los rangos de la grilla basados en los datos de entrada
    gridx = np.linspace(min(lons), max(lons), 100)
    gridy = np.linspace(min(lats), max(lats), 100)

    # Convertir listas a arrays de numpy para usar en PyKrige
    lons = np.array(lons)
    lats = np.array(lats)
    values = np.array(values)

    # Crea un objeto OrdinaryKriging con los datos y un modelo de variograma lineal
    OK = OrdinaryKriging(lons, lats, values, variogram_model=variogram_model)

    # Realiza la kriging en una grilla definida (esto es solo un ejemplo, necesitarás definir tu grilla)
    z, ss = OK.execute('grid', gridx, gridy)

    # Ahora puedes acceder a los lags y la semivarianza directamente
    lags = OK.lags
    semivariance = OK.semivariance

    # Grafica el semivariograma
    plt.figure()
    plt.plot(lags, semivariance, 'o-')
    plt.title('Semivariograma Experimental')
    plt.xlabel('Lag [unidad de distancia]')
    plt.ylabel('Semivarianza')
    plt.grid(True)

    # Guarda la grafica en un buffer en memoria
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Codifica la imagen en base64 para transferirla como JSON
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Devuelve la imagen en base64 junto con los datos de lags y semivarianza
    response = {
        'lags': OK.lags.tolist(),
        'semivariance': OK.semivariance.tolist(),
        'image_base64': image_base64
    }

    return jsonify(response)


@app.route('/kriging_contour', methods=['POST'])
def get_kriging_contour():
    data = request.json
    points = np.array(data["points"])

    lons = points[:, 0]
    lats = points[:, 1]
    values = points[:, 2]

    # Define la cuadrícula sobre la que se realizará el kriging
    grid_lon = np.linspace(min(lons), max(lons), 50)
    grid_lat = np.linspace(min(lats), max(lats), 50)

    OK = OrdinaryKriging(lons, lats, values, variogram_model='spherical')
    z, ss = OK.execute('grid', grid_lon, grid_lat)

    # Dibuja el mapa de contornos
    fig, ax = plt.subplots()
    c = ax.contourf(grid_lon, grid_lat, z, cmap="viridis")
    fig.colorbar(c, ax=ax)

    # Guarda la figura en un objeto BytesIO y luego convierte a base64
    buf = BytesIO()
    plt.savefig(buf, format="png")
    base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')

    return jsonify({"image_base64": base64_image})


@app.route('/points', methods=['GET'])
def getPoints():
    return jsonify({"points": testing_points})


if __name__ == '__main__':
    app.run(debug=True)
