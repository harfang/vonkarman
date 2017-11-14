import numpy as np
import matplotlib.pyplot as plt

"""
    u : vitesse horizontale (Ox)
    v : vitesse verticale (Oy)
    dx : pas horizontal
    dy : pas vertical
    dt : pas temporel
    Re : Reynolds
    phi : potentiel dans la projection
    p : pression
    nom_de_fonction
    i : indice Ox 
    j : indice Oy
    n : indice de temps
    Nx : nombre de points suivant Ox
    Ny : nombre de points suivant Oy
    Nt : nombre de points temporels
    T = Nt dt
    Lx = Nx dx
    Ly = Ny dy
"""

Bonjour

## Définition des constantes
Lx = 5
Ly = 1

# Taille des tableaux
Nx = 100
Ny = 500

# Taille du domaine réel
nx = Nx-2 # 2 points fantômes
ny = Ny-2

# Pas
dx = Lx/(nx-1)
dy = Ly/(ny-1)

# Constantes physiques
Re = 100
r = 0.1 # dimension de l'obstacle
if r > Ly or r > Lx:
	print("ERREUR SUR r : r > Ly or r > Lx")

Nt = 1000

## Conditions initiales
u = np.zeros(Nx, Ny)
v = np.zeros(Nx, Ny)

## Définition des fonctions
def condition_cfl(u, v, Re):
	#1. Advection
	#2. Diffusion
	#3. min
	return dt_min

def laplacien(f):
	"""Renvoie le laplacien de la fonction scalaire f"""
	laplacien_f = np.empty((NY,NX))
	dx_2 = 1/(dx)**2
	dy_2 = 1/(dy)**2
	coef0 = -2*(dx_2 + dy_2)  
	laplacien_f[1:-1,1:-1] = dx_2*(x[2:,1:-1]+x[:-2,1:-1])+dy_2*(x[1:-1, 2:]+x[1:-1,:-2])+coef0*x[1:-1,1:-1]
	return laplacien_f
	
def divergence(u, v):
	"""Renvoie la divergence du champ de vecteurs (u, v). Schéma centré (ordre 2)."""
	return div
	
def grad(f):
	"""Renvoie le gradient de f. Schéma centré (ordre 2)."""
	return grad_f_x, grad_f_y
	
def cl_objet(ustar, vstar):
	"""Modifie les tableaux pour satifsaire les conditions aux limites de la vitesse autour de l'objet"""
	pass

def cl_soufflerie(ustar, vstar):
	"""Modifie les tableaux pour satifsaire les conditions aux limites de la soufflerie"""
	pass
	
def cl_phi(phi):
	"""Modifie les tableaux pour satifsaire les conditions aux limites de la soufflerie et de l'objet"""

def solve_laplacien(div, cl_phi):
	"""Renvoie phi telle que laplacien(phi) = div, avec les conditions aux limites données par cl_phi (qui travaille directement sur les tableaux)."""
	
	return phi
	
def points_fantomes_phi(phi):
	"""Met à jour les points fantomes de phi."""
	pass
	
def points_fantomes_vitesse(u, v):
	"""Met à jour les points fantomes de la vitesse"""
	pass

## Boucle principale
for n in range(Nt):
	#Calcul du nouveau dt pour respecter les conditions CFL
	dt = condition_cfl(u, v, Re)
	
	#Calcul de l'accélération convective
	a_conv = calcul_a_conv(u, v)
	
	#Navier-Stokes
	ustar = -a_conv + (1/Re)*laplacien(u)
	vstar = -a_conv + (1/Re)*laplacien(v)
	
	#Conditions aux limites
	## Soufflerie numérique
	cl_soufflerie(ustar, vstar)
	## Bords de l'objet
	cl_objet(ustar, vstar)
	
	#Projection
	divstar = divergence(ustar, vstar)
	phi = solve_laplacien(divstar, cl_phi)
	points_fantomes_phi(phi)
	gradphi_x, gradphi_y = grad(phi) # code optimisable en réduisant le nombre de variables
	
	u = ustar - gradphi_x
	v = vstar - gradphi_y
	
	#Mise à jour des points fantômes
	points_fantomes_vitesse(u, v)
