from flask import Flask, jsonify, request
from pykrige.ok import OrdinaryKriging
import numpy as np


app = Flask(__name__)

@app.route('/semivariogram', methods=['POST'])
def get_visualizations():
    data = request.json
    lons = np.array(data["lons"])
    lats = np.array(data["lats"])
    values = np.array(data["values"])

    OK = OrdinaryKriging(lons, lats, values, variogram_model='spherical')


    return jsonify({"hs": OK.lags.tolist(), "semivariograms": OK.semivariance.tolist()})


if __name__ == '__main__':
    app.run(debug=True)