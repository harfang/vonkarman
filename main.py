# -*- coding: utf-8 -*-

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

## Definition des constantes
Lx = 1.0
Ly = 5.0

# Taille des tableaux
Nx = 10
Ny = 15

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

Nt = 5
pas_enregistrement = 1 #sauvegarde d'une image sur 30

## Conditions initiales
u = np.ones((Nx, Ny))
v = np.zeros((Nx, Ny))

## Définition de l'objet
#On place le centre de l'objet en (5r, Ly/2)
#la matrice objet renvoie une matrice pleine de 1 là où il y a l'objet et pleine de 0 là où il n'y est pas
objet=np.zeros((Nx,Ny))
for i in range(Nx):
	for j in range(Ny):
		if (i*dx-5*r)**2+(j*dy-0.5*Ly)**2 < r**2:
			objet[i][j]=1 

#objet = np.array([[1 for i in range(Nx) if (i*dx-5*r)**2+(j*dy-0.5*Ly)**2 < r**2] for j in range(Ny)])

## Définition des fonctions
def condition_cfl(u, v, Re):
	facteur_de_precaution_cfl = 0.7
	#1. Advection
#	u_max = max(np.abs(u).max(), 0.0001)
#	v_max = max(np.abs(v).max(), 0.0001)
	u_max =np.abs(u).max()
	v_max = np.abs(v).max()
	dt_adv = facteur_de_precaution_cfl * min(dx, dy)/max(u_max, v_max)


	#2. Diffusion
	#u_min = min(np.abs(u).min(), 0.001)
	#v_min = min(np.abs(v).min(), 0.001)
	#Re_min = Re*min(u_min, v_min)
	dt_diffusion = Re*min(dx**2, dy**2)
	
	#3. min
	dt_min = min(dt_diffusion, dt_adv)
	
	return dt_min

def laplacien(f):
	"""Renvoie le laplacien de la fonction scalaire f"""
	laplacien_f = np.empty((Nx, Ny))
	dx_2 = 1/(dx)**2
	dy_2 = 1/(dy)**2
	coef0 = -2*(dx_2 + dy_2)  
	laplacien_f[1:-1,1:-1] = dx_2*(f[2:,1:-1]+f[:-2,1:-1])+dy_2*(f[1:-1, 2:]+f[1:-1,:-2])+coef0*f[1:-1,1:-1]
	return laplacien_f
	
def divergence(u, v):
	"""Renvoie la divergence du champ de vecteurs (u, v). Schéma centré (ordre 2). La taille de la matrice est nx*ny"""
	div = np.empty((nx, ny))
	div = ((u[1:-1, 2:] - u[1:-1, :-2])/(dx/2) +(v[2:, 1:-1] - v[:-2, 1:-1])/(dy/2))
	return( div)
	
def grad(f):
	"""Renvoie le gradient de f. Schéma centré (ordre 2)."""
	grad_f_x = np.empty((Nx, Ny))
	grad_f_y = np.empty((Nx, Ny))
	
	grad_f_y[1:-1, :] = (f[2:, :] - f[:-2, :])/(2*dy)
	grad_f_x[:, 1:-1] = (f[:, 2:] - f[:, :-2])/(2*dx)
#	grad_f_y[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1])/(2*dy)
#	grad_f_x[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2])/(2*dx)
	return grad_f_x, grad_f_y


def construction_matrice_laplacien_2D(nx, ny):
	"""Construit et renvoie la matrice sparse du laplacien 2D"""
	dx_2 = 1/(dx)**2
	dy_2 = 1/(dy)**2
	# Axe x
	datax = [np.ones(nx), -2*np.ones(nx), np.ones(nx)]
	## Conditions aux limites : Neumann à gauche et Dirichlet à droite
	datax[2][1]     = 2.  # SF left#
	datax[0][nx-1] = 0  # SF right --> serait-ce le point fantome qui vérifie phi=0 sinon on ne peut pas calculer le laplacien au bord

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
	print(DXX2)
#	DXX2[0,:] = np.zeros(DXX2[0,:].shape)
#	DXX2[-1,:] = np.zeros(DXX2[-1,:].shape)

#	DYY2[0,:] = np.zeros(DYY2[0,:].shape)
#	DYY2[-1,:] = np.zeros(DYY2[-1,:].shape)
	
	lap2D = sp.kron( DXX2, sp.eye(ny,ny)) + sp.kron( sp.eye(nx,nx), DYY2) #sparse
	
	
	return lap2D.todense()

## Laplacien 2D
matrice_laplacien_2D = construction_matrice_laplacien_2D(nx, ny)	
	
def cl_objet(ustar, vstar):
	"""Modifie les tableaux pour satifsaire les conditions aux limites de la vitesse autour de l'objet"""
	#on multiplie ustar et vstar par une matrice pleine de 1 là où il n'y a pas l'objet et de zéros là où il y a l'objet
	#ustar=(np.ones((Nx,Ny))-objet)*ustar
	#vstar=(np.ones((Nx,Ny))-objet)*vstar
	pass

def cl_soufflerie(ustar, vstar):
	"""Modifie les tableaux pour satifsaire les conditions aux limites de la soufflerie"""
	ustar[:, :2] = 1
	vstar[:, :2] = 0
	vstar[:2, :] = 0
	vstar[Ny-3:, :] = 0
	ustar[:, Nx-1] = ustar[:, Nx-3]
	vstar[:, Nx-1] = vstar[:, Nx-3]
	
def cl_phi(phi):
	"""Modifie les tableaux pour satifsaire les conditions aux limites de l'objet"""
	phi[0,:]=phi[2,:]
	phi[-1,:]=phi[-3,:]
	phi[:,0]=phi[:,2]
	phi[:,-1]=0	
	#quelle est alors la valeur à mettre dans la dernière colonne ? on garde 0 ?
	pass

def solve_laplacien(div):
	"""Renvoie phi telle que laplacien(phi) = div, avec les conditions aux limites données par cl_phi (cl_phi travaille directement sur les tableaux). La taille de div est nx*ny, celle de lapacien son carré"""
	

	div_flat = div.flatten()
	phi_flat = np.linalg.solve(matrice_laplacien_2D, div_flat)
	phi=np.zeros((Nx,Ny))
	print('Le max de phi est ',np.max(np.abs(phi_flat)))
	phi[1:-1,1:-1] = phi_flat.reshape((nx, ny))
	cl_phi(phi)
	
	return phi
	
def calcul_a_conv(u, v):
	"""Calcul l'accélération convective u grad u"""
	
	Resu = np.empty((Nx, Ny))
	Resv = np.empty((Nx, Ny))
	
	# Matrice avec des 1 quand on va a droite, 
	# 0 a gauche ou au centre
	Mx2 = np.sign(np.sign(u[1:-1,1:-1]) + 1)
	Mx1 = 1. - Mx2

	# Matrice avec des 1 quand on va en haut, 
	# 0 en bas ou au centre
	My2 = np.sign(np.sign(v[1:-1,1:-1]) + 1.)
	My1 = 1. - My2

	# Matrices en valeurs absolues pour u et v
	au = abs(u[1:-1,1:-1]) /dx 
	av = abs(v[1:-1,1:-1]) /dy

	# Matrices des coefficients respectivement 
	# central, exterieur, meme x, meme y	 
	Cc = (1. - au) * (1. - av) 
	Ce = au * av
	Cmx = (1. - au) * av
	Cmy = (1. - av) * au

	# Calcul des matrices de resultat 
	# pour les vitesses u et v
	Resu[1:-1,1:-1] = (Cc * u[1:-1, 1:-1] +			
					   Ce * (Mx1*My1 * u[2:, 2:] + 
							 Mx1*My2 * u[:-2, 2:] +
							 Mx2*My1 * u[2:, :-2] +
							 Mx2*My2 * u[:-2, :-2]) +  
					   Cmx * (My1 * u[2:, 1:-1] +
							  My2 * u[:-2, 1:-1]) +   
					   Cmy * (Mx1 * u[1:-1, 2:] +
							  Mx2 * u[1:-1, :-2]))
	
	Resv[1:-1,1:-1] = (Cc * v[1:-1, 1:-1] +			
					   Ce * (Mx1*My1 * v[2:, 2:] + 
							 Mx1*My2 * v[:-2, 2:] +
							 Mx2*My1 * v[2:, :-2] +
							 Mx2*My2 * v[:-2, :-2]) +  
					   Cmx * (My1 * v[2:, 1:-1] +
							  My2 * v[:-2, 1:-1]) +   
					   Cmy * (Mx1 * v[1:-1, 2:] +
							  Mx2 * v[1:-1, :-2]))
	return Resu, Resv

t_simu = 0
print('t_simu=',t_simu)
print('vitesse maximale selon x =',np.max(np.abs(u)))
print('vitesse maximale selon y =',np.max(np.abs(v)))
## Boucle principale
for n in range(Nt):

	#Calcul du nouveau dt pour respecter les conditions CFL
	dt = condition_cfl(u, v, Re)
	print('dt=',dt)
	print('t_simu=',t_simu+dt)
	#Calcul de l'accélération convective
	a_conv_u, a_conv_v = calcul_a_conv(u, v)
	ustar=np.zeros((Nx,Ny))
	vstar=np.zeros((Nx,Ny))
	#Navier-Stokes
	ustar[1:-1,1:-1] = u[1:-1,1:-1] + dt*(-a_conv_u[1:-1,1:-1] + (1/Re)*laplacien(u)[1:-1,1:-1])
	vstar[1:-1,1:-1] = v[1:-1,1:-1] + dt*(-a_conv_v[1:-1,1:-1] + (1/Re)*laplacien(v)[1:-1,1:-1])
	
	#Conditions aux limites
	## Soufflerie numérique
	cl_soufflerie(ustar, vstar)
	## Bords de l'objet
	cl_objet(ustar, vstar)
	
	#Projection
	divstar = divergence(ustar, vstar)
	phi = solve_laplacien(divstar)
	gradphi_x, gradphi_y = grad(phi) # code optimisable en réduisant le nombre de variables
	
	print('vitesse maximale star selon x =',np.max(np.abs(ustar)), 'situe en', np.where(np.max(np.abs(ustar) == np.abs(ustar)))[0])
	print('vitesse maximale star selon y =',np.max(np.abs(vstar)),'situe en', np.where(np.max(np.abs(vstar) == np.abs(vstar)))[0][0])
	print('Le maximum du gradient de phi est selon x', np.max(np.abs(gradphi_x[1:-1,1:-1])))
	print('Le maximum du gradient de phi est selon y', np.max(np.abs(gradphi_y)))
	print('difference entre ',np.max(np.abs(gradphi_x-ustar)+ np.abs(gradphi_y-vstar)))
	u[1:-1,1:-1] = ustar[1:-1,1:-1] - gradphi_x[1:-1,1:-1]
	v[1:-1,1:-1] = vstar[1:-1,1:-1] - gradphi_y[1:-1,1:-1]
	cl_soufflerie(u,v)

	print('vitesse maximale selon x =',np.max(np.abs(u)))
	print('vitesse maximale selon y =',np.max(np.abs(v)))
	#Fin du calcul
	t_simu += dt
	
	## Affichage
		# Premier script : enregistrement d'une image sur pas_enregistrement
	if n%pas_enregistrement == 0:
		plt.clf()
		#plt.imshow(np.sqrt(u[1:-1,1:-1]**2+v[1:-1,1:-1]**2),origin='lower',cmap='bwr')
		plt.imshow(np.sqrt(u[1:-1,1:-1]**2),origin='lower',cmap='bwr')
		plt.colorbar()
		plt.axis('image')
		plt.savefig("{}_t={}.jpg".format( n+1, t_simu))




