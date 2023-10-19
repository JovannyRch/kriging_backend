
import matplotlib
matplotlib.use('Agg')
from flask import Flask, jsonify, request
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid
plt.style.use('ggplot')
import base64
from io import BytesIO


def generate_unique_filename(extension=".png"):
    return str(uuid.uuid4()) + extension

app = Flask(__name__)


@app.route('/health', methods=['GET'])  
def health():
    return jsonify({"status": "ok"})

@app.route('/semivariogram', methods=['POST'])
def get_semivariogram():
    data = request.json
    lons = np.array(data["lons"])
    lats = np.array(data["lats"])
    values = np.array(data["values"])

    print("==== LONS ====")
    print(lons.tolist())

    print("\n\n==== LATS ====")
    print(lats.tolist())

    print("\n\n==== VALUES ====")
    print(values.tolist())

    OK = OrdinaryKriging(lons, lats, values, variogram_model='spherical')

    plt.plot(OK.lags, OK.semivariance, '-o')
    plt.title('Semivariograma')
    plt.xlabel('Distancia')
    plt.ylabel('Semivarianza')
      

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')


    return jsonify({"hs": OK.lags.tolist(), "semivariograms": OK.semivariance.tolist(), "image_base64": image_base64})


@app.route('/kriging_contour', methods=['POST'])
def get_kriging_contour():
    
    data = request.json
    lons = np.array(data["lons"])
    lats = np.array(data["lats"])
    values = np.array(data["values"])

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



if __name__ == '__main__':
    app.run(debug=True)