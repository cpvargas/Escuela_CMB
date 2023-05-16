import numpy as np
import matplotlib
import sys
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import astropy.io.fits as fits

def crear_mapa_CMB_T(N,tamaño_pix,ell,DlTT):
    """crea una realización de un mapa del cielo CMB simulado dado un DlTT de entrada en función 
    del tamaño de píxel (tamaño_pix) requerido y el número N de píxeles en la dimensión lineal."""
    #np.random.seed(100)
    # convertir Dl a Cl
    ClTT = DlTT * 2 * np.pi / (ell*(ell+1.))
    ClTT[0] = 0. #establece el monopolo y el dipolo del espectro Cl a cero
    ClTT[1] = 0.
    
    # crea un sistema de coordenadas de espacio real en dos dimensiones (2D)
    vec_unos = np.ones(N)
    indices  = (np.arange(N)+.5 - N/2.) /(N-1.) #crear un arreglo de tamaño N entre -0.5 y +0.5
    
    #calcula la matriz del producto exterior: X[i, j] = vec_unos[i] * indices[j] para i,j
    #en range(N), que es solo N filas copias de inds - para la dimensión x
    X = np.outer(vec_unos,indices) 
    # calcula la transpuesta para la dimensión y
    Y = np.transpose(X)
    # calcula la componente radial R
    R = np.sqrt(X**2. + Y**2.)
    
    # crea un espectro de potencia CMB 2D
    pix_to_rad = (tamaño_pix/60. * np.pi/180.) # tamaño_pix en arcominutos a grados y luego de grados a radianes
    ell_factor_escala = 2. * np.pi /pix_to_rad # ahora relacionando el tamaño angular en radianes con multipolos
    ell2d = R * ell_factor_escala # crea un espacio de Fourier análogo al vector R del espacio real
    ClTT_expandido = np.zeros(int(ell2d.max())+1) 
    # hace un espectro de Cl expandido (de ceros) que llega hasta el tamaño del vector ell 2D
    ClTT_expandido[0:(ClTT.size)] = ClTT # rellena los Cls hasta el máximo del vector ClTT

    # el espectro de Cl 2D se define en el vector múltiple establecido por la escala de píxeles
    CLTT2d = ClTT_expandido[ell2d.astype(int)] 
    #plt.imshow(np.log(CLTT2d)) #función para visualizar el logaritmo del espectro 2D
    
    # ahora creamos una realización del CMB con el espectro de potencia dado en el espacio real
    arreglo_aleatoreo_para_T = np.random.normal(0,1,(N,N))
    # tomamos la FFT ya que estamos en espacio de Fourier
    FT_arreglo_aleatoreo_para_T = np.fft.fft2(arreglo_aleatoreo_para_T) 
    
    FT_2d = np.sqrt(CLTT2d) * FT_arreglo_aleatoreo_para_T  
    # hemos usado la raiz cuadrada ya que el espectro de potencia es T^2
    plt.imshow(np.real(FT_2d))

    #hacemos un gráfico del mapa simulado en espacio de Fourier, tenga en cuenta que las etiquetas de los ejes x e y
    #deben corregirse
    #Plot_CMB_Map(np.real(np.conj(FT_2d)*FT_2d*ell2d * (ell2d+1)/2/np.pi),0,np.max(np.conj(FT_2d)*FT_2d*ell2d * (ell2d+1)/2/np.pi),ell2d.max(),ell2d.max())  ###
    
    # nos movemos de vuelta desde el espacio ell al espacio real
    CMB_T = np.fft.ifft2(np.fft.fftshift(FT_2d)) 
    # nos movemos de vuelta desde espacio pixel al espacio mapa
    CMB_T = CMB_T/(tamaño_pix /60.* np.pi/180.)
    # solo queremos graficar la componente real
    CMB_T = np.real(CMB_T)

    ## devuelve el mapa
    return(CMB_T)
  ###############################

def Graficar_Mapa_CMB(Mapa_a_Graficar,c_min,c_max,X_ancho,Y_ancho):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    print("promedio del mapa:",np.mean(Mapa_a_Graficar),", media cuadrática (rms) del mapa:",np.std(Mapa_a_Graficar))
    plt.gcf().set_size_inches(10, 10)
    im = plt.imshow(Mapa_a_Graficar, interpolation='bilinear', origin='lower',cmap=cm.RdBu_r)
    im.set_clim(c_min,c_max)
    ax=plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im, cax=cax)
    #cbar = plt.colorbar()
    im.set_extent([0,X_ancho,0,Y_ancho])
    plt.ylabel('ángulo $[^\circ]$')
    plt.xlabel('ángulo $[^\circ]$')
    cbar.set_label('temperatura [uK]', rotation=270)
    
    plt.show()
    return(0)
  ###############################


def componente_Poisson(N,tamaño_pix,Numero_de_Fuentes,Amplitud_de_Fuentes):
    "crea un mapa simplificado de fuentes puntuales distribuido por Poisson"
    "toma en cuenta el numero de pixeles, el tamaño de los pixeles y el numero y amplitud de las fuentes"
    MapaFP = np.zeros([int(N),int(N)]) #Iniciamos el Mapa de Fuentes Puntuales como ceros
    i = 0.
    # vamos añadiendo fuentes con amplitud dada por la distribución de Poisson alrededor de la amplitud promedio
    while (i < Numero_de_Fuentes):
        pix_x = int(N*np.random.rand())
        pix_y = int(N*np.random.rand()) 
        MapaFP[pix_x,pix_y] += np.random.poisson(Amplitud_de_Fuentes)
        i = i + 1
    return(MapaFP)    
  ############################### 

def componente_Exponencial(N,tamaño_pix,Numero_de_Fuentes_EX,Amplitud_de_Fuentes_EX):
    "crea un mapa simplificado de fuentes puntuales distribuidas exponencialmente"
    MapaFP = np.zeros([int(N),int(N)])
    i = 0.
    # vamos añadiendo fuentes con amplitud dada por la distribución exponencial alrededor de la amplitud promedio
    while (i < Numero_de_Fuentes_EX):
        pix_x = int(N*np.random.rand()) 
        pix_y = int(N*np.random.rand()) 
        MapaFP[pix_x,pix_y] += np.random.exponential(Amplitud_de_Fuentes_EX)
        i = i + 1
    return(MapaFP)    
  ############################### 


def componente_SZ(N,tamaño_pix,Numero_de_cumulos_SZ,Amplitud_promedio_de_cumulos_SZ,
                          SZ_beta,SZ_Theta_central,hacer_grafico):
    "crea un mapa simulado de SZ"
    N=int(N)
    MapaSZ = np.zeros([N,N]) 
    catSZ = np.zeros([3,Numero_de_cumulos_SZ]) ## catalogo de fuentes SZ: X, Y, amplitud
    # crea una distribución de fuentes puntuales con amplitud variable
    i = 0
    while (i < Numero_de_cumulos_SZ):
        pix_x = int(N*np.random.rand())
        pix_y = int(N*np.random.rand())
        pix_amplitud = np.random.exponential(Amplitud_promedio_de_cumulos_SZ)*(-1.)
        catSZ[0,i] = pix_x
        catSZ[1,i] = pix_y
        catSZ[2,i] = pix_amplitud
        MapaSZ[pix_x,pix_y] += pix_amplitud
        i = i + 1
    if (hacer_grafico):
        hist,bin_bordes = np.histogram(MapaSZ,bins = 50,range=[MapaSZ.min(),-10])
        plt.figure(figsize=(10,10))
        plt.semilogy(bin_bordes[0:-1],hist)
        plt.xlabel('amplitud de fuentes [$\mu$K]')
        plt.ylabel('numero de pixeles')
        plt.show()
    
    # crea una función beta
    beta = funcion_beta(int(N),tamaño_pix,SZ_beta,SZ_Theta_central)
    
    
    # convoluciona la función beta con las fuentes puntuales para obtener el mapa SZ
    # NOTA: puedes volver a la introducción para practicar con convoluciones!
    TF_beta = np.fft.fft2(np.fft.fftshift(beta))    # Transformada de Fourier de la función beta
    TF_MapaSZ = np.fft.fft2(np.fft.fftshift(MapaSZ))  # Transformada de Fourier del mapa puntual SZ
    MapaSZ = np.fft.fftshift(np.real(np.fft.ifft2(TF_beta*TF_MapaSZ))) # Convolución 
    
    # retorna el mapa SZ, junto con el catalogo
    return(MapaSZ,catSZ)    
  ############################### 

def funcion_beta(N,tamaño_pix,SZ_beta,SZ_Theta_central):
  # crea la función beta

    N=int(N)
    unos = np.ones(N)
    indices  = (np.arange(N)+.5 - N/2.) * tamaño_pix
    X = np.outer(unos,indices)
    Y = np.transpose(X)
    # calcula la misma función de espacio-real R que en el caso de FP
    R = np.sqrt(X**2. + Y**2.)
    
    beta = (1 + (R/SZ_Theta_central)**2.)**((1-3.*SZ_beta)/2.)

    # retorna el mapa de función beta
    return(beta)
  ############################### 

def mapa_convolucionado_con_haz_gaussiano(N,tamaño_pix,tamaño_haz_fwhm,Mapa):
    """convoluciona un mapa con un patrón de haz gaussiano. 
    NOTA: tamaño_pix y tamaño_haz_fwhm deben estar en las mismas unidades"""

    # crea una Gaussiana 2D
    gaussiana = hacer_haz_gaussiano_2d(N,tamaño_pix,tamaño_haz_fwhm)
  
    # realiza la convolución
    TF_gaussiana = np.fft.fft2(np.fft.fftshift(gaussiana)) # primero usamos shift para que quede centrado
    TF_Mapa = np.fft.fft2(np.fft.fftshift(Mapa)) #shift en el mapa también
    mapa_convolucionado = np.fft.fftshift(np.real(np.fft.ifft2(TF_gaussiana*TF_Mapa)))
    
    # retorna el mapa convolucionado
    return(mapa_convolucionado)
  ###############################   

def hacer_haz_gaussiano_2d(N,tamaño_pix,tamaño_haz_fwhm):  
     # hacer el sistema de coordenadas 2d
    N=int(N)
    unos = np.ones(N)
    indices  = (np.arange(N)+.5 - N/2.) * tamaño_pix
    X = np.outer(unos,indices)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2.)
  
    # hacer la gaussiana 2d
    sigma_haz = tamaño_haz_fwhm / np.sqrt(8.*np.log(2))
    gaussiana = np.exp(-.5 *(R/sigma_haz)**2.)
    gaussiana = gaussiana / np.sum(gaussiana)
    # retorna la gaussiana

    return(gaussiana)
  ###############################  

def hacer_mapa_de_ruido(N,tamaño_pix,nivel_de_ruido_blanco,nivel_de_ruido_atmosferico,nivel_de_ruido_1sobref):
    "hace una realización de ruido instrumental, atmosférico y 1/f"
    
    ## hacer un mapa de ruido blanco
    ruido_blanco = np.random.normal(0,1,(N,N)) * nivel_de_ruido_blanco/tamaño_pix
 
    ## hacer un mapa de ruido atmosperico
    ruido_atmosferico = 0.
    if (nivel_de_ruido_atmosferico != 0):
        unos = np.ones(N)
        indices  = (np.arange(N)+.5 - N/2.) 
        X = np.outer(unos,indices)
        Y = np.transpose(X)
        R = np.sqrt(X**2. + Y**2.) * tamaño_pix /60. ## angulos relativos a 1 grado
        mag_k = 2 * np.pi/(R+.01)  ## 0.01 es un factor de regularización
        ruido_atmosferico = np.fft.fft2(np.random.normal(0,1,(N,N)))
        ruido_atmosferico  = np.fft.ifft2(atmospheric_noise * np.fft.fftshift(mag_k**(5/3.)))
        ruido_atmosferico = ruido_atmosferico * nivel_de_ruido_atmosferico/tamaño_pix

    ## hace un mapa 1/f, a lo largo de una sola dirección para ilustrar las bandas 
    ruido_1sobref = 0.
    if (nivel_de_ruido_1sobref != 0): 
        unos = np.ones(N)
        indices  = (np.arange(N)+.5 - N/2.) 
        X = np.outer(unos,indices) * tamaño_pix /60. ## angulos relativos a 1 grado
        kx = 2 * np.pi/(X+.01) ## 0.01 es un factor de regularización
        ruido_1sobref = np.fft.fft2(np.random.normal(0,1,(N,N)))
        ruido_1sobref = np.fft.ifft2(ruido_1sobref * np.fft.fftshift(kx))* nivel_de_ruido_1sobref/tamaño_pix

    ## retorna el mapa de ruido
    mapa_ruido = np.real(ruido_blanco + ruido_atmosferico + ruido_1sobref)
    return(mapa_ruido)
  ###############################

def Mapa_Filtrado(Mapa,N,N_mascara):
    ## configurar las coordenadas x, y, r para la generación de máscaras
    unos = np.ones(N) # N es el tamaño del mapa
    indices  = (np.arange(N)+.5 - N/2.) 

    X = np.outer(unos,indices)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2.)  ## angulos relativos a 1 grado
    
    ## crea la mascara
    mascara = np.ones([N,N])
    mascara[np.where(np.abs(X) < N_mascara)]  = 0
    
    ## aplica el filtro en espacio de Fourier
    MapaF = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Mapa)))
    MapaF_filtrado = MapaF * mascara
    Mapa_filtrado = np.real(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(MapaF_filtrado))))
    
    ## retorna el mapa filtrado
    return(Mapa_filtrado)


def apply_filter(Map,filter2d):
    ## apply the filter in fourier space
    FMap = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Map)))
    FMap_filtered = FMap * filter2d
    Map_filtered = np.real(np.fft.fftshift(np.fft.fft2(FMap_filtered)))
    
    ## return the output
    return(Map_filtered)


def ventana_coseno(N):
    "crea una ventana de coseno para apodizar y evitar efectos de bordes en la transformada de Fourier 2d"
    # hace un sistema de coordenadas 2d
    unos = np.ones(N)
    indices = (np.arange(N)+.5 - N/2.)/N *np.pi ## por ejemplo, va de -pi/2 a pi/2
    X = np.outer(unos,indices)
    Y = np.transpose(X)
  
    # hace un mapa de ventana
    mapa_ventana = np.cos(X) * np.cos(Y)
   
    # retorna el mapa de ventana
    return(mapa_ventana)
  ###############################


def average_N_spectra(spectra,N_spectra,N_ells):
    avgSpectra = np.zeros(N_ells)
    rmsSpectra = np.zeros(N_ells)
    
    # calcuate the average spectrum
    i = 0
    while (i < N_spectra):
        avgSpectra = avgSpectra + spectra[i,:]
        i = i + 1
    avgSpectra = avgSpectra/(1. * N_spectra)
    
    #calculate the rms of the spectrum
    i =0
    while (i < N_spectra):
        rmsSpectra = rmsSpectra +  (spectra[i,:] - avgSpectra)**2
        i = i + 1
    rmsSpectra = np.sqrt(rmsSpectra/(1. * N_spectra))
    
    return(avgSpectra,rmsSpectra)




def calcular_espectro_2d(Mapa,delta_ell,ell_max,tamaño_pix,N,Mapa2=None):
    """calcula el espectro de potencia de un mapa 2d usando la transformada rápida de Fourier FFT,
    elevando al cuadrado, y promediando azimutalmente"""

    #hacer un sistema de coordenadas 2d ell
    N=int(N)
    unos = np.ones(N)
    indices  = (np.arange(N)+.5 - N/2.) /(N-1.)
    kX = np.outer(unos,indices) / (tamaño_pix/60. * np.pi/180.)
    kY = np.transpose(kX)
    K = np.sqrt(kX**2. + kY**2.)
    ell_factor_escala = 2. * np.pi 
    ell2d = K * ell_factor_escala
    
    # crea un arreglo para guardar los resultados del espectro de potencia
    N_bins = int(ell_max/delta_ell)
    ell_arreglo = np.arange(N_bins)
    CL_arreglo = np.zeros(N_bins)
    
    # obtener la transformada de Fourier 2d del mapa
    TF_Mapa = np.fft.ifft2(np.fft.fftshift(Mapa))
    if Mapa2 is None: TF_Mapa2 = TF_Mapa.copy()
    else: TF_Mapa2 = np.fft.ifft2(np.fft.fftshift(Mapa2))
    
    EP_Mapa = np.fft.fftshift(np.real(np.conj(TF_Mapa) * TF_Mapa2))

    # completa el espectros
    i = 0
    while (i < N_bins):
        ell_arreglo[i] = (i + 0.5) * delta_ell
        inds_in_bin = ((ell2d >= (i* delta_ell)) * (ell2d < ((i+1)* delta_ell))).nonzero()
        CL_arreglo[i] = np.mean(EP_Mapa[inds_in_bin])
        i = i + 1


    CL_arreglo_nuevo = CL_arreglo[~np.isnan(CL_arreglo)]
    ell_arreglo_nuevo = ell_arreglo[~np.isnan(CL_arreglo)]
    # retorna el espectro de potencia y los bins en ell
    return(ell_arreglo_nuevo,CL_arreglo_nuevo*np.sqrt(tamaño_pix /60.* np.pi/180.)*2.)
