
import matplotlib
matplotlib.use('Agg')
from flask import Flask, jsonify, request
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid
plt.style.use('ggplot')
""" import base64
from io import BytesIO """


def generate_unique_filename(extension=".png"):
    return str(uuid.uuid4()) + extension

app = Flask(__name__)

@app.route('/semivariogram', methods=['POST'])
def get_semivariogram():
    data = request.json
    lons = np.array(data["lons"])
    lats = np.array(data["lats"])
    values = np.array(data["values"])

    OK = OrdinaryKriging(lons, lats, values, variogram_model='spherical')

    plt.plot(OK.lags, OK.semivariance, '-o')
    plt.title('Semivariograma')
    plt.xlabel('Distancia')
    plt.ylabel('Semivarianza')
    
    fileName = generate_unique_filename()
    semivariogram_path = os.path.join("static", fileName)
    plt.savefig(semivariogram_path)
    

    """  buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8') """


    return jsonify({"hs": OK.lags.tolist(), "semivariograms": OK.semivariance.tolist(), "image_url": fileName})


if __name__ == '__main__':
    app.run(debug=True)