import numpy as np
import skgstat as skg
import matplotlib.pyplot as plt

data = [
    [205958218, -1003412843, 28.2],
    [205953995, -1003420983, 28.6],
    [20595551, -1003413033, 16.8],
    [205968874, -1003427943, 17.8],
    [205975481, -1003363917, 17.6],
    [205942038, -1003405744, 22.4],
    [205951775, -1003396479, 22.6],
    [2059565, -1003412083, 25.9],
    [205965882, -1003384912, 14.8],
    [205965784, -1003384853, 22.4],
    [206007563, -1003390777, 21.9],
    [205972327, -1003387707,  22],
    [20.5970135, -1003412704, 16.9],
    [205973858, -100341391, 23.8],
    [206000873, -1003394943, 27.7],
    [205990706, -1003412689, 17]
]

coordinates = np.array(data)[:, 0:2]
values = np.array(data)[:, 2]

print(coordinates)
print(values)

V = skg.Variogram(coordinates=coordinates, values=values)


bins = V.bins
experimental_semivariance = V.experimental

# Graficar el semivariograma experimental
plt.figure(figsize=(6, 4))
plt.plot(bins, experimental_semivariance, 'o-')
plt.title('Semivariograma Experimental')
plt.xlabel('Distancia')
plt.ylabel('Semivarianza')
plt.grid(True)
plt.show()


print(V)
