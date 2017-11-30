# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:09:33 2017

@author: utilisateur
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse as sp

"""
	u : vitesse horizontale (Ox) matrice Ny par Nx
	v : vitesse verticale (Oy) matrice Ny par Nx
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


Plan du code :
	définition des constantes et des objets globaux
	Les opérateurs basiques :
		gradient(scalaire Ny,Nx) --> renvoie un objet de taille Ny*Nx MAIS NE PAS USER LES GHOSTS POINTS
		divergence (U,V de taille Ny,Nx) --> renvoie un objet de taille Ny*Nx MAIS NE PAS USER LES GHOST POINTS
		laplacien (scalaire Ny,Nx) --> renvoie un objet de taille Ny*Nx MAIS NE PAS USER LES GHOSTS POINTS
	Les conditions limites :
		Conditions CFL
		Conditions limites de la soufflerie
		Conditions limites dûes à l'objet
		Conditions limites sur phi (multiplicateur de Lagrange <--> pression)
		Mise à jour des points fantômes
	Les grosses fonctions :
		accélération covective
		construction de la matrice laplacien
		résolution de l'équation de Poisson
"""


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
	
Nt = 5 #durée de la simulation
pas_enregistrement = 1 #sauvegarde d'une image sur pas_enregistrement

#Définition de l'objet 
#A FAIRE !!! matrice pleine de 1 là où il est et pleine de zéros là où il n'y est pas




"""
---------------------------------------------------------------------------------
|										|
|				LES OPÉRATEURS BASIQUES				|
|										|
---------------------------------------------------------------------------------
"""


def divergence(u, v):
	"""Renvoie la divergence du champ de vecteurs (u, v). Schéma centré (ordre 2). La taille de la matrice est Ny*Nx"""
	div = np.zeros((Ny, Nx))
	div[1:-1, 1:-1] = ((u[1:-1, 2:] - u[1:-1, :-2])/(dx*2) +(v[2:, 1:-1] - v[:-2, 1:-1])/(dy*2))
	return( div)

def grad(f):
	"""Renvoie le gradient de f. Schéma centré (ordre 2)."""
	grad_f_x = np.empty((Ny, Nx))
	grad_f_y = np.empty((Ny, Nx))
	grad_f_y[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1])/(2*dy)
	grad_f_x[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2])/(2*dx)
	return grad_f_x, grad_f_y

def laplacien(f):
	"""Renvoie le laplacien de la fonction scalaire f"""
	laplacien_f = np.empty((Ny, Nx))
	dx_2 = 1/(dx)**2
	dy_2 = 1/(dy)**2
	coef0 = -2*(dx_2 + dy_2)  
	laplacien_f[1:-1,1:-1] = dy_2*(f[2:,1:-1]+f[:-2,1:-1])+dx_2*(f[1:-1, 2:]+f[1:-1,:-2])+coef0*f[1:-1,1:-1]
	return laplacien_f



"""
---------------------------------------------------------------------------------
|										|
|				LES CONDITIONS LIMITES				|
|										|
---------------------------------------------------------------------------------
"""



	
def condition_cfl(u, v, Re):
	"""Il faut que le pas de temps de la simulation numérique soit plus petit que le temps caractéristique de diffusion et d'advection"""
	facteur_de_precaution_cfl = 0.7
	#1. Advection
	u_max =np.abs(u).max()
	v_max = np.abs(v).max()
	dt_adv = facteur_de_precaution_cfl * min(dx, dy)/max(u_max, v_max)

	#2. Diffusion
	dt_diffusion = Re*min(dx**2, dy**2)
	
	#3. min
	dt_min = min(dt_diffusion, dt_adv)
	return dt_min

def cl_soufflerie(ustar, vstar):
	"""Modifie les tableaux pour satifsaire les conditions aux limites de la soufflerie"""
	ustar[:, :2] = 1 #la vitesse est de 1 en entrée
	vstar[:, :2] = 0 #la vitesse v est nulle en entrée
	ustar[0,1:-1]= ustar[2,1:-1] #en haut la contrainte est nulle
	vstar[:2, :] = 0 #en haut, la vitesse normale est nulle
	ustar[-1,1:-1]= ustar[-3,1:-1] #en haut la contrainte est nulle
	vstar[Ny-2:, :] = 0 #en bas, la vitesse normale est nulle
	ustar[:, Nx-1] = ustar[:, Nx-3] #dérivée de u par rapport à x nulle à la sortie
	vstar[:, Nx-1] = vstar[:, Nx-3] #dérivée de v par rapport à x nulle à la sortie
	
def cl_objet(ustar, vstar):
	"""Modifie les tableaux pour satifsaire les conditions aux limites de la vitesse autour de l'objet"""
	pass

def cl_phi(phi):
	"""Modifie les tableaux pour satifsaire les conditions aux limites de l'objet"""
	phi[0,:]=phi[2,:]
	phi[-1,:]=phi[-3,:]
	phi[:,0]=phi[:,2]
	phi[:,-1]=0
	#quelle est alors la valeur à mettre dans la dernière colonne ? on garde 0 ?
	pass

def points_fantome_vitesse(u,v):
	u[:, 0] = 1 #la vitesse est de 1 en entrée
	v[:, 0] = 0 #la vitesse v est nulle en entrée
	u[0,:]= u[2,:] #en haut la contrainte est nulle
	v[0, :] = 0 #en haut, la vitesse normale est nulle
	u[-1,:]= u[-3,:] #en haut la contrainte est nulle
	v[-1, :] = 0 #en bas, la vitesse normale est nulle
	u[:, Nx-1] = u[:, Nx-3] #dérivée de u par rapport à x nulle à la sortie
	v[:, Nx-1] = v[:, Nx-3] #dérivée de v par rapport à x nulle à la sortie
	



"""
---------------------------------------------------------------------------------
|										|
|				LES GROSSES FONCTIONS				|
|										|
---------------------------------------------------------------------------------
"""



def calcul_a_conv(u, v):
	"""Calcul l'accélération convective u grad u"""
	
	Resu = np.empty((Ny, Nx))
	Resv = np.empty((Ny, Nx))
	
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

def construction_matrice_laplacien_2D(nx, ny):
	"""Construit et renvoie la matrice sparse du laplacien 2D"""
	dx_2 = 1/(dx)**2
	dy_2 = 1/(dy)**2
	offsets = np.array([-1,0,1])                    

	# Axe x
	datax = [np.ones(nx), -2*np.ones(nx), np.ones(nx)]
	DXX = sp.dia_matrix((datax,offsets), shape=(nx,nx)) * dx_2
	DXX2 = DXX.todense()
	## Conditions aux limites : Neumann à gauche et Dirichlet à droite
	DXX2[0,1]     = 2.  # SF left : au lieu d'un 1 on met un 2 à la deuxième colonne première ligne
	DXX2[0,-1] = 0  # SF right 
	# Axe Y
	datay = [np.ones(ny), -2*np.ones(ny), np.ones(ny)] 
	DYY = sp.dia_matrix((datay,offsets), shape=(ny,ny)) * dy_2	
	DYY2 = DYY.todense()  
	## Conditions aux limites : Neumann 
	DYY2[0,1]     = 2.   # en haut
	DYY2[-1, ny-2] = 2.  # en bas
	lap2D = sp.kron( sp.eye(Ny,Ny) , DXX2 ) + sp.kron(DYY2, sp.eye(Nx,Nx)) #sparse
	return lap2D.todense()
	
def solve_laplacien(div):
	"""Renvoie phi telle que laplacien(phi) = div, avec les conditions aux limites données par cl_phi (cl_phi travaille directement sur les tableaux). La taille de div est nx*ny, celle de lapacien son carré"""
	div_flat = div.flatten()
	phi_flat = np.linalg.solve(matrice_laplacien_2D, div_flat)
	phi=np.zeros((Nx,Ny))
	print('Le max de phi est ',np.max(np.abs(phi_flat)))
	phi[1:-1,1:-1] = phi_flat.reshape((nx, ny))
	cl_phi(phi)
	return phi


matrice_laplacien_2D = construction_matrice_laplacien_2D(nx, ny)	


