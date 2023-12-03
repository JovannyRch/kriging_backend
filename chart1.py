import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

# Supongamos que tienes los siguientes datos
x = np.array([1, 2, 3, 4, 5])   # Datos del eje X
y = np.array([5, 4, 3, 2, 1])   # Datos del eje Y
# Datos de 'Z' (pueden representar densidad, incidencia, etc.)
z = np.array([10, 20, 30, 40, 50])

# Crear un gráfico de dispersión con colores basados en 'Z'
plt.scatter(x, y, c=z, cmap='viridis')  # 'cmap' define el mapa de colores
plt.colorbar()  # Muestra la barra de colores
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.title('Gráfico de Dispersión con Densidad de Incidencias')

# Guardar el gráfico en un buffer temporal
buf = BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)

# Convertir la imagen en el buffer a una cadena base64
image_base64 = base64.b64encode(buf.read()).decode('utf-8')

# `image_base64` es ahora una representación en base64 de la imagen
print(image_base64)
