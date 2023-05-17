import numpy as np
import matplotlib.pyplot as plt
# Usamos algunas cosas que aprendimos antes
from cmb_modules import calculate_2d_spectrum,make_CMB_T_map

# Necesitamos cargar los espectros de la teoría
def get_theory():
    ells,tt,_,_,pp,_ = np.loadtxt("CAMB_fiducial_cosmo_scalCls.dat",unpack=True)
    TCMB2 = 7.4311e12
    ckk = pp/4./TCMB2
    ucltt = tt / ells/(ells+1.)*2.*np.pi
    ells2,lcltt = np.loadtxt("CMB_fiducial_totalCls.dat",unpack=True,usecols=[0,1])
    lcltt = lcltt / ells2/(ells2+1.)*2.*np.pi
    lcltt = lcltt[:len(ells)]
    return ells,ucltt,lcltt,ckk

def get_lensed(patch_deg_width,pix_size,ells,ucltt,clkk):
    
    # Número de píxeles en cada dirección
    N = int(patch_deg_width*60./pix_size)
    # Luego generamos un mapa CMB sin lentes como un campo aleatorio gaussiano como aprendimos antes
    DlTT = ucltt*ells*(ells+1.)/2./np.pi
    unlensed = make_CMB_T_map(N,pix_size,ells,DlTT)
    # También necesitamos un mapa de convergencia de lentes (kappa)
    DlKK = clkk*ells*(ells+1.)/2./np.pi
    kappa = make_CMB_T_map(N,pix_size,ells,DlKK)
    # obtenemos las coordenadas de Fourier
    ly,lx,modlmap = get_ells(N,pix_size)
    # Ahora podemos lensear nuestro mapa de entrada no lenseado
    lensed = lens_map(unlensed,kappa,modlmap,ly,lx,N,pix_size)
    return N,lensed,kappa,ly,lx,modlmap

def lens_map(imap,kappa,modlmap,ly,lx,N,pix_size):
    # Primero convertimos la convergencia de lente en potencial de lente
    phi = kappa_to_phi(kappa,modlmap,return_fphi=True)
    # Luego tomamos su gradiente para obtener el campo de deflexión.
    grad_phi = gradient(phi,ly,lx)
    # Luego calculamos las posiciones desplazadas cambiando las posiciones físicas usando las desviaciones
    pos = posmap(N,pix_size) + grad_phi
    # Convertimos las posiciones desplazadas en números de píxeles desplazados 
    # porque scipy no sabe de distancias fisicas
    pix = sky2pix(pos, N,pix_size)
    # Preparamos un arreglo vacío para que contenga el resultado
    omap = np.empty(imap.shape, dtype= imap.dtype)
    # Luego le decimos a scipy que calcule los valores del mapa lenseado de entrada en las posiciones 
    # fraccionarias desplazadas por interpolación y cuadricule eso en el mapa lenseado final
    from scipy.ndimage import map_coordinates
    map_coordinates(imap, pix, omap, order=5, mode='wrap')
    return omap

# Esta función necesita conocer las coordenadas de Fourier del mapa.
def get_ells(N,pix_size):
    # Esta función devuelve números de onda de Fourier para una cuadrícula cartesiana cuadrada
    N=int(N)
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) /(N-1.)
    ell_scale_factor = 2. * np.pi 
    lx = np.outer(ones,inds) / (pix_size/60. * np.pi/180.) * ell_scale_factor
    ly = np.transpose(lx)
    modlmap = np.sqrt(lx**2. + ly**2.)
    return ly,lx,modlmap

# Necesitamos convertir kappa a phi
def kappa_to_phi(kappa,modlmap,return_fphi=False):
    return filter_map(kappa,kmask(2./modlmap/(modlmap+1.),modlmap,ellmin=2))

# función de enmascaramiento de espacio de Fourier que será útil
def kmask(filter2d,modlmap,ellmin=None,ellmax=None):
    # Aplica una máscara de multipolos mínimos y máximos a un filtro
    if ellmin is not None: filter2d[modlmap<ellmin] = 0
    if ellmax is not None: filter2d[modlmap>ellmax] = 0
    return filter2d

# Para hacer eso, también necesitamos saber en general cómo filtrar un mapa.
def filter_map(Map,filter2d):
    FMap = np.fft.fftshift(np.fft.fft2(Map))
    FMap_filtered = FMap * filter2d
    Map_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(FMap_filtered)))
    return Map_filtered

# También necesitamos calcular un gradiente
# Lo hacemos en el espacio de Fourier
def gradient(imap,ly,lx):
    # Filtra el mapa por (i ly, i lx) para obtener un degradado
    return np.stack([filter_map(imap,ly*1j),filter_map(imap,lx*1j)])

# También necesitábamos el mapa de posiciones físicas
def posmap(N,pix_size):
    pix    = np.mgrid[:N,:N]
    return pix2sky(pix,N,pix_size)

# Para eso, necesitamos poder convertir índices de píxeles en posiciones del cielo.
def pix2sky(pix,N,pix_size):
    py,px = pix
    dec = np.deg2rad((py - N//2 - 0.5)*pix_size/60.)
    ra = np.deg2rad((px - N//2 - 0.5)*pix_size/60.)
    return np.stack([dec,ra])

# Finalmente, para la operación de lente, también necesitábamos convertir las posiciones físicas 
# del cielo en índices de píxeles, que es justo lo contrario de lo anterior
def sky2pix(pos,N,pix_size):
    dec,ra = np.rad2deg(pos)*60.
    py = dec/pix_size + N//2 + 0.5
    px = ra/pix_size + N//2 + 0.5
    return np.stack([py,px])
