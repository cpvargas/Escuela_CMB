## variables para configurar el tamaño del mapa
N = 2**10  # este es el número de píxeles en una dimensión lineal
            ## dado que estamos usando muchas FFT, esto debería ser un factor de 2 ^ N
tamaño_pix  = 0.5 # tamaño de un píxel en minutos de arco
N_iteraciones = 16
## variables para configurar los graficos del mapa
c_min = -400  # Mínimo para la barra de color
c_max = 400   # Máximo para la barra de color
X_ancho = N*tamaño_pix/60.  # ancho de mapa horizontal en grados
Y_ancho = N*tamaño_pix/60.  # ancho de mapa vertical en grados

tamaño_haz_fwhm = 1.25

Numero_de_Fuentes  = 5000
Amplitud_de_Fuentes = 200
Numero_de_Fuentes_EX = 50
Amplitud_de_Fuentes_EX = 1000
Numero_de_Cumulos_SZ = 500
Amplitud_promedio_de_cumulos_SZ = 50
SZ_beta = 0.86
SZ_Theta_central = 1.0


nivel_de_ruido_blanco = 10.
nivel_de_ruido_atmosferico = 0.5*0.
nivel_de_ruido_1sobref = 0.

#### parámetros para configurar el espectro
delta_ell = 50.
ell_max = 5000.
