# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:09:33 2017

@author: utilisateur
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import os

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



"""
---------------------------------------------------------------------------------
|																				|
|				CONSTANTES ET OBJETS											|
|																				|
---------------------------------------------------------------------------------
"""

class VonKarman():

	def __init__(self, Lx = 30.0, Ly = 7.0, Nx = 150, Ny=35, Nt=1200, r = 0.35, Re = 100, pas_enregistrement = 5):
		## Definition des constantes
		self.Lx = Lx
		self.Ly = Ly
		
		# Taille des tableaux
		self.Nx = Nx
		self.Ny = Ny

		# Taille du domaine réel
		self.nx = self.Nx-2 # 2 points fantômes
		self.ny = self.Ny-2

		# Pas
		self.dx = self.Lx/(self.nx-1)
		self.dy = self.Ly/(self.ny-1)
		#self.dt = 0.001

		# Constantes physiques
		self.Re = Re
		self.r = r # dimension de l'obstacle
		if 4*r > Ly or 10*r > Lx:
			print("ERREUR SUR r : r > Ly or r > Lx")
	
		self.Nt = Nt #durée de la simulation
		self.pas_enregistrement = pas_enregistrement #sauvegarde d'une image sur pas_enregistrement


		## Conditions initiales
		self.u = np.zeros((self.Ny, self.Nx))
		self.v = np.zeros((self.Ny, self.Nx))
		self.phi = np.zeros((self.Ny, self.Nx))
		self.ustar = np.zeros((self.Ny, self.Nx))
		self.vstar = np.zeros((self.Ny, self.Nx))

		#Définition de l'objet 
		## Définition de l'objet
		#On place le centre de l'objet en -8r, Ly/2)
		#la matrice objet renvoie une matrice pleine de 0 là où il y a l'objet et pleine de 1 là où il n'y est pas
		self.objet=np.ones((Ny,Nx))
		for i in range(Ny): 
			for j in range(Nx):
				if (j*self.dx-15*self.r)**2+(i*self.dy-0.5*(self.Ly+2*self.dy))**2 < self.r**2:
					self.objet[i][j]=0
					
		# Affichage
		self.dirname = ""
		
		# Laplacien
		self.construction_matrice_laplacien_2D()

	"""
	---------------------------------------------------------------------------------
	|																				|
	|				PREPARATION AFFICHAGE ET ENREGISTREMENT							|
	|																				|
	---------------------------------------------------------------------------------
	"""
	
	def create_dir(self):
		"""Création du dossier pour l'enregistrement"""
		self.dirname = os.path.dirname(os.path.realpath(__file__))+"/"+str(time.strftime("%Y-%m-%d - %X"))
		if not os.path.exists(self.dirname):
			os.makedirs(self.dirname)
		else:
			timestamp = time.time()
			self.dirname += str(timestamp-int(timestamp))[1:]
			os.makedirs(self.dirname)
	
		os.chdir(self.dirname) # change le répertoire de travail

		np.savetxt("parameters.txt", np.array([self.Lx, self.Ly, self.Nx, self.Ny, self.Nt, self.r, self.Re]).transpose(), header = "Lx \t Ly \t Nx \t Ny \t Nt \t r \t Re \n", newline="\t")
	
		## Préparation de la création d'un gif animé
		self.command_string = "convert *.jpg -delay 10 -loop 0 \\\n"

		""" ATTENTION LA LIGNE QUI SUIT VA PROVOQUER UNE ERREUR SI LA NORME DE LA VITESSE DEPASSE vmax"""
		## Normalisation des couleurs
		self.color_norm = matplotlib.colors.Normalize(vmin=0.,vmax=2.)
		self.color_norm_w = matplotlib.colors.Normalize(vmin=-8,vmax=8)

	"""
	---------------------------------------------------------------------------------
	|																				|
	|				LES OPÉRATEURS BASIQUES											|
	|																				|
	---------------------------------------------------------------------------------
	"""


	def divergence(self, u, v):
		"""Renvoie la divergence du champ de vecteurs (u, v). Schéma centré (ordre 2). La taille de la matrice est Ny*Nx"""
		div = np.zeros((self.Ny, self.Nx))
		div[1:-1, 1:-1] = ((u[1:-1, 2:] - u[1:-1, :-2])/(self.dx*2) +(v[2:, 1:-1] - v[:-2, 1:-1])/(self.dy*2))
		return div

	def grad(self, f):
		"""Renvoie le gradient de f. Schéma centré (ordre 2)."""
		grad_f_x = np.empty((self.Ny, self.Nx))
		grad_f_y = np.empty((self.Ny, self.Nx))
		grad_f_y[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1])/(2*self.dy)
		grad_f_x[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2])/(2*self.dx)
		return grad_f_x, grad_f_y

	def laplacien(self, f):
		"""Renvoie le laplacien de la fonction scalaire f"""
		laplacien_f = np.empty((self.Ny, self.Nx))
		dx_2 = 1/(self.dx)**2
		dy_2 = 1/(self.dy)**2
		coef0 = -2*(dx_2 + dy_2)  
		laplacien_f[1:-1,1:-1] = dy_2*(f[2:,1:-1]+f[:-2,1:-1])+dx_2*(f[1:-1, 2:]+f[1:-1,:-2])+coef0*f[1:-1,1:-1]
		return laplacien_f

	"""
	---------------------------------------------------------------------------------
	|																				|
	|				AUTRES OBSERVABLES												|
	|																				|
	---------------------------------------------------------------------------------
	"""

	def w(self):
		"""Renvoie la norme du vecteur vorticité"""
		u_x, u_y = self.grad(self.u)
		v_x, v_y = self.grad(self.v)
		
		return v_x-u_y

	"""
	---------------------------------------------------------------------------------
	|																				|
	|				LES CONDITIONS LIMITES											|
	|																				|
	---------------------------------------------------------------------------------
	"""
	
	def condition_cfl(self):
		"""Il faut que le pas de temps de la simulation numérique soit plus petit que le temps caractéristique de diffusion et d'advection"""
		facteur_de_precaution_cfl = 0.7
		#1. Advection
		u_max = np.abs(self.u).max()
		v_max = np.abs(self.v).max()
		dt_adv = facteur_de_precaution_cfl * min(self.dx, self.dy)/max(u_max, v_max, 1)

		#2. Diffusion
		dt_diffusion = self.Re*min(self.dx**2, self.dy**2)
	
		#3. min
		dt_min = min(dt_diffusion, dt_adv)
		self.dt = dt_min

	def cl_soufflerie(self):
		"""Modifie les tableaux pour satifsaire les conditions aux limites de la soufflerie"""
		self.ustar[:, 0] = 1 #la vitesse est de 1 en entrée
		self.ustar[:, 1] = 1 #la vitesse est de 1 en entrée
		self.vstar[:, :2] = 0 #la vitesse v est nulle en entrée
		self.ustar[0,1:-1]= self.ustar[2,1:-1] #en haut la contrainte est nulle
		self.vstar[:2, :] = 0 #en haut, la vitesse normale est nulle
		self.ustar[-1,1:-1]= self.ustar[-3,1:-1] #en bas la contrainte est nulle
		self.vstar[self.Ny-2:, :] = 0 #en bas, la vitesse normale est nulle
		self.ustar[:, self.Nx-1] = self.ustar[:, self.Nx-3] #dérivée de u par rapport à x nulle à la sortie
		self.vstar[:, self.Nx-1] = self.vstar[:, self.Nx-3] #dérivée de v par rapport à x nulle à la sortie
	#	ustar[Ny//2,:2]=3
	#	ustar[Ny//2+1,:2]=3
	
	def cl_objet(self):
		"""Modifie les tableaux pour satifsaire les conditions aux limites de la vitesse autour de l'objet"""
		#self.ustar *= self.objet
		#self.vstar *= self.objet
		pass

	def cl_phi(self):
		"""Modifie les tableaux pour satifsaire les conditions aux limites de l'objet"""
		self.phi[0,:] = self.phi[2,:]
		self.phi[-1,:] = self.phi[-3,:]
		self.phi[:,0] = self.phi[:,2]
		self.phi[:,-1] = 0
		#quelle est alors la valeur à mettre dans la dernière colonne ? on garde 0 ?

	def points_fantome_vitesse(self, u, v):
		u[:, 0] = 1 #la vitesse est de 1 en entrée
		v[:, 0] = 0 #la vitesse v est nulle en entrée
		u[0,:]= u[2,:] #en haut la contrainte est nulle
		v[0, :] = 0 #en haut, la vitesse normale est nulle
		u[-1,:]= u[-3,:] #en haut la contrainte est nulle
		v[-1, :] = 0 #en bas, la vitesse normale est nulle
		u[:, self.Nx-1] = u[:, self.Nx-3] #dérivée de u par rapport à x nulle à la sortie
		v[:, self.Nx-1] = v[:, self.Nx-3] #dérivée de v par rapport à x nulle à la sortie

	"""
	---------------------------------------------------------------------------------
	|																				|
	|				LES GROSSES FONCTIONS											|
	|																				|
	---------------------------------------------------------------------------------
	"""

	def calcul_a_conv(self):
		"""Calcul l'accélération convective u grad u"""
		#problème : il utilise une autre convention que la notre c'est-à-dire que pour lui x est le premier indice. 

		Resu = np.empty((self.Ny, self.Nx))
		Resv = np.empty((self.Ny, self.Nx))

		# Matrice avec des 1 quand on va a droite, 
		# 0 a gauche ou au centre
		Mx2 = np.sign(np.sign(self.u[1:-1,1:-1]) + 1)
		Mx1 = 1. - Mx2

		# Matrice avec des 1 quand on va en haut, 
		# 0 en bas ou au centre
		My2 = np.sign(np.sign(self.v[1:-1,1:-1]) + 1.)
		My1 = 1. - My2

		# Matrices en valeurs absolues pour u et v
		au = abs(self.u[1:-1,1:-1])*self.dt/self.dx
		av = abs(self.v[1:-1,1:-1])*self.dt/self.dy

		# Matrices des coefficients qui va pondérer la vitesse des points respectivement
		# central, exterieur, meme x, meme y	 
		Cc = (1. - au) * (1. - av) 
		Ce = au * av
		Cmx = (1. - au) * av
		Cmy = (1. - av) * au

		# Calcul des matrices de resultat 
		# pour les vitesses u et v
		Resu[1:-1,1:-1] = (Cc * self.u[1:-1, 1:-1] +			
						   Ce * (Mx1*My1 * self.u[2:, 2:] + 
								 Mx1*My2 * self.u[:-2, 2:] +
								 Mx2*My1 * self.u[2:, :-2] +
								 Mx2*My2 * self.u[:-2, :-2]) +  
						   Cmx * (My1 * self.u[2:, 1:-1] +
								  My2 * self.u[:-2, 1:-1]) +   
						   Cmy * (Mx1 * self.u[1:-1, 2:] +
								  Mx2 * self.u[1:-1, :-2]))
	
		Resv[1:-1,1:-1] = (Cc * self.v[1:-1, 1:-1] +			
						   Ce * (Mx1*My1 * self.v[2:, 2:] + 
								 Mx1*My2 * self.v[:-2, 2:] +
								 Mx2*My1 * self.v[2:, :-2] +
								 Mx2*My2 * self.v[:-2, :-2]) +  
						   Cmx * (My1 * self.v[2:, 1:-1] +
								  My2 * self.v[:-2, 1:-1]) +   
						   Cmy * (Mx1 * self.v[1:-1, 2:] +
								  Mx2 * self.v[1:-1, :-2]))
		return Resu, Resv

	def construction_matrice_laplacien_2D(self):
		"""Construit et renvoie la matrice sparse du laplacien 2D"""
		dx_2 = 1/(self.dx)**2
		dy_2 = 1/(self.dy)**2
		offsets = np.array([-1,0,1])                    

		# Axe x
		datax = [np.ones(self.nx), -2*np.ones(self.nx), np.ones(self.nx)]
		DXX = sp.dia_matrix((datax,offsets), shape=(self.nx,self.nx)) * dx_2
		DXX2 = DXX.todense()
		## Conditions aux limites : Neumann à gauche et Dirichlet à droite
		DXX2[0,1]     = 2.  # SF left : au lieu d'un 1 on met un 2 à la deuxième colonne première ligne
		DXX2[0,-1] = 0  # SF right 
		# Axe Y
		datay = [np.ones(self.ny), -2*np.ones(self.ny), np.ones(self.ny)] 
		DYY = sp.dia_matrix((datay,offsets), shape=(self.ny,self.ny)) * dy_2	
		DYY2 = DYY.todense()  
		## Conditions aux limites : Neumann 
		DYY2[0,1]     = 2.   # en haut
		DYY2[-1, self.ny-2] = 2.  # en bas
		self.matrice_laplacien_2D = sp.kron(sp.eye(self.ny, self.ny), DXX2) + sp.kron(DYY2, sp.eye(self.nx, self.nx)) #sparse
		#self.matrice_laplacien_2D = lap2D.todense()
	
	def solve_laplacien(self, div):
		"""Renvoie phi telle que laplacien(phi) = div, avec les conditions aux limites données par cl_phi (cl_phi travaille directement sur les tableaux). La taille de div est nx*ny, celle de lapacien son carré"""
		div_flat = div.flatten()
		phi_flat = linalg.spsolve(self.matrice_laplacien_2D, div_flat)
		self.phi = np.zeros((self.Ny,self.Nx))
		self.phi[1:-1,1:-1] = phi_flat.reshape((self.ny,self.nx))

	"""
	---------------------------------------------------------------------------------
	|																				|
	|				BOUCLE PRINCIPALE												|
	|																				|
	---------------------------------------------------------------------------------
	"""
	def main_loop(self):
		t_simu = 0
		t_simu_precedemment_enregistre = 0

		t_i = time.time()
		
		self.create_dir()

		## Boucle principale
		for n in range(self.Nt):

			#Calcul du nouveau dt pour respecter les conditions CFL
			self.condition_cfl()
		
			#Calcul de l'accélération convective
			Resu, Resv = self.calcul_a_conv()
			#Navier-Stokes
			self.ustar[1:-1,1:-1] = Resu[1:-1,1:-1]+self.dt*self.laplacien(self.u)[1:-1,1:-1]/self.Re
			self.vstar[1:-1,1:-1] = Resv[1:-1,1:-1]+self.dt*self.laplacien(self.v)[1:-1,1:-1]/self.Re
	
			#Conditions aux limites
			## Soufflerie numérique
			self.cl_soufflerie()
			## Bords de l'objet
			#self.cl_objet()
	
			#Projection
			divstar = self.divergence(self.ustar, self.vstar)
			self.solve_laplacien(divstar[1:-1,1:-1])
			gradphi_x, gradphi_y = self.grad(self.phi) # code optimisable en réduisant le nombre de variables
	
			self.u[1:-1,1:-1] = self.ustar[1:-1,1:-1] - gradphi_x[1:-1,1:-1]
			self.v[1:-1,1:-1] = self.vstar[1:-1,1:-1] - gradphi_y[1:-1,1:-1]

			
			# Mise à jour des points fantomes
			self.points_fantome_vitesse(self.u, self.v)
			self.points_fantome_vitesse(self.ustar, self.vstar)
			
			#Fin du calcul
			t_simu += self.dt
	
			## Affichage
				# Premier script : enregistrement d'une image sur pas_enregistrement
			if n%self.pas_enregistrement == 0:
				print(n, 'sur', self.Nt)
				plt.clf()
				#plt.imshow(1-objet[1:-1,1:-1], origin = 'lower', cmap='binary', alpha = 0.)
				#x = 0
				#plt.plot(self.u[:,x], color="blue", label="u")
				#plt.plot(self.v[:,x], color="green", label="v")
				#plt.legend(loc="best")
				## Affichage de la norme de la vitesse
				plt.imshow(np.sqrt(self.u[1:-1,1:-1]**2+self.v[1:-1,1:-1]**2), origin='lower', cmap='afmhot', interpolation = 'none', norm = self.color_norm)
			
				## Affichage de la vorticité
		#		plt.imshow(self.w()[1:-1,1:-1], origin='lower', cmap='bwr', interpolation = 'none', norm = self.color_norm_w)
				
		#		plt.imshow(self.phi, origin='lower', cmap='bwr', interpolation = 'none')#, norm = self.color_norm)
			
				plt.colorbar()

				plt.axis('image')
				if n<10:		
					plt.savefig("image000{}_t={}_Re={}.jpg".format(n+1, t_simu, self.Re), dpi = 60)
				elif n<100:
					plt.savefig("image00{}_t={}_Re={}.jpg".format(n+1, t_simu, self.Re), dpi = 60)
				elif n<1000:
					plt.savefig("image0{}_t={}_Re={}.jpg".format(n+1, t_simu, self.Re), dpi = 60)
				else:
					plt.savefig("image{}_t={}_Re={}.jpg".format(n+1, t_simu, self.Re), dpi = 60)
			
				self.command_string += "\\( -clone {} -set delay {} \\) -swap {} +delete \\\n".format(n//self.pas_enregistrement, (t_simu-t_simu_precedemment_enregistre)*10, n//self.pas_enregistrement)
				t_simu_precedemment_enregistre = t_simu
	
		self.command_string += " movie.gif"
		os.system(self.command_string)
		print("Simulation effectuée en {} s. Données enregistrées dans {}.".format(time.time()-t_i, self.dirname))

simu = VonKarman(Re = 50, Nt = 600, Nx = 300, Ny = 70, pas_enregistrement = 5)
simu.main_loop()
