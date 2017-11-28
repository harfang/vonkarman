import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse as sp


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

date_simulation = time.time()

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
pas_enregistrement = 30 #sauvegarde d'une image sur 30

## Conditions initiales
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))


## Definition de l'objet
#On place le centre de l'objet en (5r, Ly/2)
#la matrice objet renvoie une matrice pleine de 1 là où il y a l'objet et pleine de 0 là où il n'y est pas
#objet=np.zeros(Nx,Ny)
#for i in range(Nx):
#	for j in range(Ny):
#		if (i*dx-5*r)**2+(j*dy-0.5*Ly)**2 < r**2:
#			objet[i][j]=1 

objet = np.array([[1 if (i*dx-5*r)**2+(j*dy-0.5*Ly)**2 < r**2 for i in range(Nx)] for j in range(Ny)])

## Laplacien 2D
matrice_laplacien_2D = construction_matrice_laplacien_2D(Nx, Ny)

## Definition des fonctions
def condition_cfl(u, v, Re):
	facteur_de_precaution_cfl = 0.7
	#1. Advection
	u_max = max(np.abs(u).max(), 0.001)
	v_max = max(np.abs(v).max(), 0.001)
	dt_adv = facteur_de_precaution_cfl * min(dx, dy)/max(u_max, v_max)
	
	#2. Diffusion
	u_min = min(np.abs(u).min(), 0.001)
	v_min = min(np.abs(v).min(), 0.001)
	Re_min = Re*min(u_min, v_min)
	dt_diffusion = Re_min*min(dx**2, dy**2)
	
	#3. min
	dt_min = min(dt_diffusion, dt_adv)
	
	return dt_min

def laplacien(f):
	"""Renvoie le laplacien de la fonction scalaire f"""
	laplacien_f = np.empty((Nx, Ny))
	dx_2 = 1/(dx)**2
	dy_2 = 1/(dy)**2
	coef0 = -2*(dx_2 + dy_2)  
	laplacien_f[1:-1,1:-1] = dx_2*(x[2:,1:-1]+x[:-2,1:-1])+dy_2*(x[1:-1, 2:]+x[1:-1,:-2])+coef0*x[1:-1,1:-1]
	return laplacien_f
	
def divergence(u, v):
	"""Renvoie la divergence du champ de vecteurs (u, v). Schéma centré (ordre 2)."""
	div = np.empty((Nx, Ny))
	div[1:-1,1:-1] = ((u[1:-1, 2:] - u[1:-1, :-2])/(dx/2) +(v[2:, 1:-1] - v[:-2, 1:-1])/(dy/2))
	return div
	
def grad(f):
	"""Renvoie le gradient de f. Schéma centré (ordre 2)."""
	grad_f_x = np.empty((Nx, Ny))
	grad_f_y = np.empty((Nx, Ny))
	
	grad_f_x[1:-1, :] = (f[2:, :] - f[:-2, :])/(2*dx)
	grad_f_y[:, 1:-1] = (f[:, 2:] - f[:, :-2])/(2*dy)
	
	return grad_f_x, grad_f_y
	
def construction_matrice_laplacien_2D(nx, ny):
	"""Construit et renvoie la matrice sparse du laplacien 2D"""
	dx_2 = 1/(dx)**2
	dy_2 = 1/(dy)**2
	# Axe x
	datax = [np.ones(nx), -2*np.ones(nx), np.ones(nx)]
		
	## Conditions aux limites : Neumann à gauche et Dirichlet à droite
	datax[2][1]     = 2.  # SF left
	datax[0][nx-2] = 2.  # SF right

#	# Axe Y
	datay = [np.ones(ny), -2*np.ones(ny), np.ones(ny)] 
#	  
	## Conditions aux limites : Neumann 
	datay[2][1]     = 2.  # SF low
	datay[0][ny-2] = 2.  # SF top

	# Construction de la matrice sparse
	offsets = np.array([-1,0,1])                    
	DXX = sp.dia_matrix((datax,offsets), shape=(nx,nx)) * dx_2
	DYY = sp.dia_matrix((datay,offsets), shape=(ny,ny)) * dy_2
	
	DXX2 = DXX.todense()
	DYY2 = DYY.todense()
	
#	DXX2[0,:] = np.zeros(DXX2[0,:].shape)
#	DXX2[-1,:] = np.zeros(DXX2[-1,:].shape)

#	DYY2[0,:] = np.zeros(DYY2[0,:].shape)
#	DYY2[-1,:] = np.zeros(DYY2[-1,:].shape)
	
	lap2D = sp.kron( DXX2, sp.eye(Ny,Ny)) + sp.kron( sp.eye(Nx,Nx), DYY2) #sparse
	
	
	return lap2D
	
def cl_objet(ustar, vstar):
	"""Modifie les tableaux pour satifsaire les conditions aux limites de la vitesse autour de l'objet"""
	
	#on multiplie ustar et vstar par une matrice pleine de 1 là où il n'y a pas l'objet et de zéros là où il y a l'objet
	ustar=(np.ones(Nx,Ny)-objet)*ustar
	vstar=(np.ones(Nx,Ny)-objet)*vstar 

def cl_soufflerie(ustar, vstar):
	"""Modifie les tableaux pour satifsaire les conditions aux limites de la soufflerie"""
	pass
	
def cl_phi(phi):
	"""Modifie les tableaux pour satifsaire les conditions aux limites de la soufflerie et de l'objet"""
	pass

def solve_laplacien(div, cl_phi):
	"""Renvoie phi telle que laplacien(phi) = div, avec les conditions aux limites données par cl_phi (cl_phi travaille directement sur les tableaux)."""
	
	return phi
	
def points_fantomes_phi(phi):
	"""Met à jour les points fantomes de phi."""
	pass
	
def points_fantomes_vitesse(u, v):
	"""Met à jour les points fantomes de la vitesse"""
	pass

t_simu = 0
"""
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
	
	#Fin du calcul
	t_simu += dt
	
	## Affichage
	
	# Premier script : enregistrement d'une image sur pas_enregistrement
	if n%pas_enregistrement == 0:
		plt.clf()
		plt.imshow(np.sqrt(u[1:-1,1:-1]**2+v[1:-1,1:-1]**2),origin='lower',cmap='bwr')
		plt.axis('image')
		plt.savefig("{}_{}_t={}.jpg".format(date_simulation, n, t_simu))
"""
Nt = 1000
pas_enregistrement = 30 #sauvegarde d'une image sur 30

## Conditions initiales
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))


## Definition de l'objet
#On place le centre de l'objet en (5r, Ly/2)
#la matrice objet renvoie une matrice pleine de 1 là où il y a l'objet et pleine de 0 là où il n'y est pas
#objet=np.zeros(Nx,Ny)
#for i in range(Nx):
#	for j in range(Ny):
#		if (i*dx-5*r)**2+(j*dy-0.5*Ly)**2 < r**2:
#			objet[i][j]=1 

objet = np.array([[1 if (i*dx-5*r)**2+(j*dy-0.5*Ly)**2 < r**2 for i in range(Nx)] for j in range(Ny)])

## Laplacien 2D
matrice_laplacien_2D = construction_matrice_laplacien_2D(Nx, Ny)

## Definition des fonctions
def condition_cfl(u, v, Re):
	facteur_de_precaution_cfl = 0.7
	#1. Advection
	u_max = max(np.abs(u).max(), 0.001)
	v_max = max(np.abs(v).max(), 0.001)
	dt_adv = facteur_de_precaution_cfl * min(dx, dy)/max(u_max, v_max)
	
	#2. Diffusion
	u_min = min(np.abs(u).min(), 0.001)
	v_min = min(np.abs(v).min(), 0.001)
	Re_min = Re*min(u_min, v_min)
	dt_diffusion = Re_min*min(dx**2, dy**2)
	
	#3. min
	dt_min = min(dt_diffusion, dt_adv)
	
	return dt_min

def laplacien(f):
	"""Renvoie le laplacien de la fonction scalaire f"""
	laplacien_f = np.empty((Nx, Ny))
	dx_2 = 1/(dx)**2
	dy_2 = 1/(dy)**2
	coef0 = -2*(dx_2 + dy_2)  
	laplacien_f[1:-1,1:-1] = dx_2*(x[2:,1:-1]+x[:-2,1:-1])+dy_2*(x[1:-1, 2:]+x[1:-1,:-2])+coef0*x[1:-1,1:-1]
	return laplacien_f
	
def divergence(u, v):
	"""Renvoie la divergence du champ de vecteurs (u, v). Schéma centré (ordre 2)."""
	div = np.empty((Nx, Ny))
	div[1:-1,1:-1] = ((u[1:-1, 2:] - u[1:-1, :-2])/(dx/2) +(v[2:, 1:-1] - v[:-2, 1:-1])/(dy/2))
	return div
	
def grad(f):
	"""Renvoie le gradient de f. Schéma centré (ordre 2)."""
	grad_f_x = np.empty((Nx, Ny))
	grad_f_y = np.empty((Nx, Ny))
	
	grad_f_x[1:-1, :] = (f[2:, :] - f[:-2, :])/(2*dx)
	grad_f_y[:, 1:-1] = (f[:, 2:] - f[:, :-2])/(2*dy)
	
	return grad_f_x, grad_f_y
	
def construction_matrice_laplacien_2D(Nx, Ny):
	"""Construit et renvoie la matrice sparse du laplacien 2D"""
	dx_2 = 1/(dx)**2
	dy_2 = 1/(dy)**2
	# Axe x
	datax = [np.ones(Nx), -2*np.ones(Nx), np.ones(Nx)]
		
#	## Conditions aux limites : Neumann 
#	datax[2][1]     = 2.  # SF left
#	datax[0][Nx-2] = 2.  # SF right

#	# Axe Y
#	datay = [np.ones(Ny), -2*np.ones(Ny), np.ones(Ny)] 
#	  
#	## Conditions aux limites : Neumann 
#	datay[2][1]     = 2.  # SF low
#	datay[0][Ny-2] = 2.  # SF top

	# Construction de la matrice sparse
	offsets = np.array([-1,0,1])                    
	DXX = sp.dia_matrix((datax,offsets), shape=(Nx,Nx)) * dx_2
	DYY = sp.dia_matrix((datay,offsets), shape=(Ny,Ny)) * dy_2
	
	DXX2 = DXX.todense()
	DYY2 = DYY.todense()
	
	DXX2[0,:] = np.zeros(DXX2[0,:].shape)
	DXX2[-1,:] = np.zeros(DXX2[-1,:].shape)

	DYY2[0,:] = np.zeros(DYY2[0,:].shape)
	DYY2[-1,:] = np.zeros(DYY2[-1,:].shape)
	
	lap2D = sp.kron(sp.eye(Ny,Ny), DXX2) + sp.kron(DYY2, sp.eye(Nx,Nx)) #sparse
	#Il faut maintenant prendre en compte les conditions limites
	#Par exemple, les CL imposent phi_2,j=phi_0,j ainsi on peut le voir comme une application linéaire qui
	#envoie le vecteur phi_2,j sur phi_0,j et phi_2,j, l'image de phi_0,j est le vecteur nul et ceci pour tout j
	#Il suffira ensuite de multiplier la matrice précédente par cette matrice et on aura les conditions limites en haut
	matrice = np.eye(Nx*Ny)
	#On la modifie pour satisfaire les conditions limites en haut
	for i in  range(Nx):
		matrice[i,i]=0
		matrice[i+2*Nx,i]=1
	#Maintenant pour satisfaire les conditions limites en bas
	for i in range(Nx*Ny-Nx, Nx*Ny):
		matrice[i,i]=0
		matrice[i-2*Nx,i]=1
	#On fait les conditions limites de gauche
	for j in range(Ny):
		matrice[Nx*j,Nx*j]=0
		matrice[Nx*j+2,Nx*j]=1 		
	#et les conditions à droite
	for j in range(Ny):
		matrice[j*Nx-2,j*Nx-2]=0

	return lap2D
	
def cl_objet(ustar, vstar):
	"""Modifie les tableaux pour satifsaire les conditions aux limites de la vitesse autour de l'objet"""
	
	#on multiplie ustar et vstar par une matrice pleine de 1 là où il n'y a pas l'objet et de zéros là où il y a l'objet
	ustar=(np.ones(Nx,Ny)-objet)*ustar
	vstar=(np.ones(Nx,Ny)-objet)*vstar 

def cl_soufflerie(ustar, vstar):
	"""Modifie les tableaux pour satifsaire les conditions aux limites de la soufflerie"""
	pass
	
def cl_phi(phi):
	"""Modifie les tableaux pour satifsaire les conditions aux limites de la soufflerie et de l'objet"""
	pass

def solve_laplacien(div, cl_phi):
	"""Renvoie phi telle que laplacien(phi) = div, avec les conditions aux limites données par cl_phi (cl_phi travaille directement sur les tableaux)."""
	
	return phi
	
def points_fantomes_phi(phi):
	"""Met à jour les points fantomes de phi."""
	pass
	
def points_fantomes_vitesse(u, v):
	"""Met à jour les points fantomes de la vitesse"""
	pass

t_simu = 0
"""
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
	
	#Fin du calcul
	t_simu += dt
	
	## Affichage
	
	# Premier script : enregistrement d'une image sur pas_enregistrement
	if n%pas_enregistrement == 0:
		plt.clf()
		plt.imshow(np.sqrt(u[1:-1,1:-1]**2+v[1:-1,1:-1]**2),origin='lower',cmap='bwr')
		plt.axis('image')
		plt.savefig("{}_{}_t={}.jpg".format(date_simulation, n, t_simu))
"""
