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
import scipy.signal
import scipy.fftpack as fft
import math

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
|										|										|
|				CONSTANTES ET OBJETS				|
|										|										|
---------------------------------------------------------------------------------
"""

class VonKarman():

	def __init__(self, Lx = 30.0, Ly = 7.0, Nx = 150, Ny=35, Nt=1200, r = 0.35, Re = 100, pas_enregistrement = 5, dpi = 100, vitesse_video = 5, colorant_initial= []):
		## Definition des constantes
		self.Lx = Lx
		self.Ly = Ly
		self.dt=100
		
		# Taille des tableaux
		self.Nx = int(Nx)
		self.Ny = int(Ny)

		# Taille du domaine réel
		self.nx = self.Nx-2 # 2 points fantômes
		self.ny = self.Ny-2
		
		# Pas
		self.dx = self.Lx/(self.nx-1)
		self.dy = self.Ly/(self.ny-1)

		# Constantes physiques
		self.Re = Re
		self.r = r # dimension de l'obstacle
		if 4*r > Ly or 10*r > Lx:
			print("ERREUR SUR r : r > Ly or r > Lx")
	
		self.Nt = int(Nt) #durée de la simulation
		self.pas_enregistrement = int(pas_enregistrement) #sauvegarde d'une image sur pas_enregistrement
		self.vitesse_video = vitesse_video #on peut faire aller la vidé vitesse_video fois plus vite car sinon trop long

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
		self.objet=np.ones((self.Ny, self.Nx))
		for i in range(self.Ny): 
			for j in range(self.Nx):
				if (j*self.dx-15*self.r)**2+(i*self.dy-0.5*(self.Ly+2*self.dy))**2 < self.r**2:
					self.objet[i][j]=0
		# Affichage
		self.dirname = ""
		self.root_dir = ""
		self.dpi = int(dpi)
		# Laplacien
		self.construction_matrice_laplacien_2D()
		self.construction_matrice_laplacien_2D_vitesse()

	"""
	---------------------------------------------------------------------------------
	|										|										|
	|				LES OPÉRATEURS BASIQUES				|
	|										|										|
	---------------------------------------------------------------------------------
	"""


	def divergence(self, u, v):
		"""Renvoie la divergence du champ de vecteurs (u, v). Schéma centré (ordre 2). La taille de la matrice est Ny*Nx. L'axe des y est aussi orienté vers le bas."""
		div = np.zeros((self.Ny, self.Nx))
		div[1:-1, 1:-1] = ((u[1:-1, 2:] - u[1:-1, :-2])/(self.dx*2) +(v[2:, 1:-1] - v[:-2, 1:-1])/(self.dy*2))
		return div

	def grad(self, f):
		"""Renvoie le gradient de f. Schéma centré (ordre 2). L'axe des y est orienté vers le bas."""
		grad_f_x = np.zeros((self.Ny, self.Nx))
		grad_f_y = np.zeros((self.Ny, self.Nx))

		grad_f_y[1:-1, :] = (f[2:, :] - f[:-2, :])/(2*self.dy)
		grad_f_x[:, 1:-1] = (f[:, 2:] - f[:, :-2])/(2*self.dx)
		return grad_f_x, grad_f_y

	def laplacien(self, f):
		"""Renvoie le laplacien de la fonction scalaire f"""
		laplacien_f = np.zeros((self.Ny, self.Nx))
		dx_2 = 1/(self.dx)**2
		dy_2 = 1/(self.dy)**2
		coef0 = -2*(dx_2 + dy_2)  
		laplacien_f[1:-1,1:-1] = dy_2*(f[2:,1:-1]+f[:-2,1:-1])+dx_2*(f[1:-1, 2:]+f[1:-1,:-2])+coef0*f[1:-1,1:-1]
		return laplacien_f

	"""
	---------------------------------------------------------------------------------
	|										|										|
	|				AUTRES OBSERVABLES				|
	|										|										|
	---------------------------------------------------------------------------------
	"""
	
	@property
	def w(self):
		"""Renvoie la norme du vecteur vorticité"""
		u_x, u_y = self.grad(self.u)
		v_x, v_y = self.grad(self.v)
		
		return v_x-u_y
	def colo(self):
		"""Cette fonction met à jour la position de chaque particule de colorant 
	en calculant sa vitesse et donc sa position en t+dt. Colorant est une liste attention !! """
		retrait=[]
		for n in range((len(self.colorant))):
			i = int(self.colorant[n][0])
			j = int(self.colorant[n][1])
			if j > self.nx-3:
				retrait.append(n)
			else:
				deltai = self.colorant[n][0]-i
				deltaj = self.colorant[n][1]-j
				self.colorant[n][0]+=self.dt*((1-deltai)*(1-deltaj)*self.v[i,j]+(1-deltaj)*deltai*self.v[i+1,j]+deltaj*(1-deltai)*self.v[i,j+1]+deltai*deltaj*self.v[i+1,j+1])/self.dy
				self.colorant[n][1]+=self.dt*((1-deltai)*(1-deltaj)*self.u[i,j]+(1-deltaj)*deltai*self.u[i+1,j]+deltaj*(1-deltai)*self.u[i,j+1]+deltai*deltaj*self.u[i+1,j+1])/self.dx
		for k in retrait:
			self.colorant.pop(k)
		for k in self.colorant_initial:
			self.colorant.append([k[0],k[1]])
		return(self.colorant)

	"""
	---------------------------------------------------------------------------------
	|										|											|
	|				LES CONDITIONS LIMITES				|
	|										|										|
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
		dt_diffusion = facteur_de_precaution_cfl * self.Re*min(self.dx**2, self.dy**2)/10 
	
		#3. minimum entre les deux
		dt_min = min(dt_diffusion, dt_adv)
		self.dt = dt_min

	def cl_soufflerie(self):
		"""Modifie les tableaux pour satifsaire les conditions aux limites de la soufflerie"""
		self.ustar[:, 1] = 1 #self.u0 #la vitesse est de 1 en entrée
		self.vstar[:, 1] = 0 #la vitesse v est nulle en entrée
		self.vstar[1, :] = 0
		self.vstar[-2, :] = 0
			
	def cl_objet(self):
		"""Modifie les tableaux pour satifsaire les conditions aux limites de la vitesse autour de l'objet"""
		self.ustar *= self.objet
		self.vstar *= self.objet

	def cl_phi(self):
		"""Modifie les tableaux pour satifsaire les conditions aux limites de l'objet"""
		self.phi[0,:] = self.phi[2,:]
		self.phi[-1,:] = self.phi[-3,:]
		self.phi[:,0] = self.phi[:,2]
		self.phi[:,-1] = -self.phi[:,-2]

	def points_fantome_vitesse(self, u, v):
		u[:, 0] = 2*u[:, 1] - u[:, 2] #la vitesse est de 1 en entrée
		v[:, 0] = 2*v[:, 1] - v[:, 2] #la vitesse v est nulle en entrée
		u[0, :] = u[2, :] #en haut la contrainte est nulle
		v[0, :] = 2*v[1, :] - v[2, :] #en haut, la vitesse normale est nulle
		u[-1,:] = u[-3,:] #en haut la contrainte est nulle
		v[-1, :] = 2*v[-2, :] - v[-3, :] #en bas, la vitesse normale est nulle
		u[:, -1] = u[:, -2] #dérivée de u par rapport à x nulle à la sortie
		v[:, -1] = v[:, -2] #dérivée de v par rapport à x nulle à la sortie



	"""
	---------------------------------------------------------------------------------
	|										|										|
	|				LES GROSSES FONCTIONS				|
	|										|										|
	---------------------------------------------------------------------------------
	"""

	def calcul_a_conv(self):
		"""permet de calculer l'accélération convective"""

		Resu = np.zeros((self.Ny, self.Nx))
		Resv = np.zeros((self.Ny, self.Nx))

		# Matrice avec des 1 quand on va a droite, 
		# 0 a gauche ou au centre
		Mx2 = np.sign(np.sign(self.u[1:-1,1:-1]) + 1.)
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
		DXX2[0,1]     = 2.*dx_2  # SF left : au lieu d'un 1 on met un 2 à la deuxième colonne première ligne
		DXX2[-1,-1] = -3*dx_2  # SF right 

		# Axe Y
		datay = [np.ones(self.ny), -2*np.ones(self.ny), np.ones(self.ny)] 
		DYY = sp.dia_matrix((datay,offsets), shape=(self.ny,self.ny)) * dy_2	
		DYY2 = DYY.todense()  
		## Conditions aux limites : Neumann 
		DYY2[0,1]     = 2.*dy_2  # en haut
		DYY2[-1, self.ny-2] = 2.*dy_2  # en bas
		self.matrice_laplacien_2D = sp.kron(sp.eye(self.ny, self.ny), DXX2) + sp.kron(DYY2, sp.eye(self.nx, self.nx)) #sparse
	
	def solve_laplacien(self, div):
		"""Renvoie phi telle que laplacien(phi) = div, avec les conditions aux limites données par cl_phi (cl_phi travaille directement sur les tableaux). La taille de div est nx*ny, celle de lapacien son carré"""
		div_flat = div.flatten()
		phi_flat = linalg.spsolve(self.matrice_laplacien_2D, div_flat)
		self.phi = np.zeros((self.Ny,self.Nx))
		self.phi[1:-1,1:-1] = phi_flat.reshape((self.ny,self.nx))		
	
	def construction_matrice_laplacien_2D_vitesse(self):
		dx_2 = 1/((self.dx)**2)
		dy_2 = 1/((self.dy)**2)
		offsets = np.array([-1,0,1])                    
		# Axe x
		datax = [np.ones(self.nx), -2*np.ones(self.nx), np.ones(self.nx)]
		UXX = sp.dia_matrix((datax,offsets), shape=(self.nx,self.nx)) * dx_2
		UXX2 = UXX.todense()
		VXX = sp.dia_matrix((datax,offsets), shape=(self.nx,self.nx)) * dx_2
		VXX2 = VXX.todense()
		## Conditions aux limites : u=1 et v=0 à gauche, dérivée nulle à droite
		UXX2[0,0] = 0
		UXX2[0,1] = 0 
		UXX2[-1,-1] = -1.* dx_2
		VXX2[0,0] = 0
		VXX2[0,1] = 0 
		VXX2[-1,-1] = -1.* dx_2
		# Axe Y
		datay = [np.ones(self.ny), -2*np.ones(self.ny), np.ones(self.ny)] 
		UYY = sp.dia_matrix((datay,offsets), shape=(self.ny,self.ny)) * dy_2	
		UYY2 = UYY.todense() 
		VYY = sp.dia_matrix((datay,offsets), shape=(self.ny,self.ny)) * dy_2	
		VYY2 = VYY.todense() 
		## Conditions aux limites : v nulle et dérivée par rapport à y de u nulle 
		UYY2[0,1] = 2.*dy_2  # en haut
		VYY2[0,0] = 0
		VYY2[0,1] = 0
		UYY2[-1, self.ny-2] = 2.*dy_2  # en bas
		VYY2[-1, self.ny-2] = 0
		VYY2[-1, self.ny-1] = 0
		self.matrice_laplacien_2D_u = sp.kron(sp.eye(self.ny, self.ny), UXX2) + sp.kron(UYY2, sp.eye(self.nx, self.nx)) #sparse
		self.matrice_laplacien_2D_v = sp.kron(sp.eye(self.ny, self.ny), VXX2) + sp.kron(VYY2, sp.eye(self.nx, self.nx)) #sparse
		
	def navier_stokes(self, Resu, Resv):
		"""effectue l'équation de N-S. Pour de bas Reynolds, on  résout l'équation en implicite. """
		if self.Re> 100:
			self.ustar = np.zeros((self.Ny, self.Nx))
			self.vstar = np.zeros((self.Ny, self.Nx))
			self.ustar[1:-1,1:-1] = Resu[1:-1,1:-1]+self.dt*self.laplacien(self.u)[1:-1,1:-1]/self.Re
			self.vstar[1:-1,1:-1] = Resv[1:-1,1:-1]+self.dt*self.laplacien(self.v)[1:-1,1:-1]/self.Re
			return(self.ustar, self.vstar)
		else:
			u_flat = Resu[1:-1,1:-1].flatten()
			v_flat = Resv[1:-1,1:-1].flatten()
			operateur_u = sp.kron(sp.eye(self.ny,self.ny), sp.eye(self.nx, self.nx))-self.dt*self.matrice_laplacien_2D_u/self.Re
			operateur_v = sp.kron(sp.eye(self.ny,self.ny), sp.eye(self.nx, self.nx))-self.dt*self.matrice_laplacien_2D_v/self.Re
			ustar_flat = linalg.spsolve(operateur_u, u_flat)
			vstar_flat = linalg.spsolve(operateur_v, v_flat)
			self.ustar = np.zeros((self.Ny, self.Nx))
			self.vstar = np.zeros((self.Ny, self.Nx))
			self.ustar[1:-1,1:-1] = ustar_flat.reshape((self.ny, self.nx))
			self.vstar[1:-1,1:-1] = vstar_flat.reshape((self.ny, self.nx))
			return(self.ustar, self.vstar)


	"""
	---------------------------------------------------------------------------------
	|										|										|
	|				AFFICHAGE ET ENREGISTREMENT			|
	|										|										|
	---------------------------------------------------------------------------------
	"""

		
	def create_working_dir(self):
		self.root_dir = os.path.dirname(os.path.realpath(__file__))
		self.dirname = self.root_dir+"/"+str(time.strftime("%Y-%m-%d - %X"))+" - Re = "+str(self.Re)+" - Nt = "+str(self.Nt)
		if not os.path.exists(self.dirname): os.makedirs(self.dirname)
		else:
			timestamp = time.time()
			self.dirname += str(timestamp-int(timestamp))[1:]
			os.makedirs(self.dirname)
	
		os.chdir(self.dirname) # change le répertoire de travail

		np.savetxt("parameters.txt", np.array([self.Lx, self.Ly, self.Nx, self.Ny, self.Nt, self.r, self.Re]).transpose(), header = "Lx \t Ly \t Nx \t Ny \t Nt \t r \t Re \n", newline="\t")

	def callback_gif_start(self):
		"""Création du dossier pour l'enregistrement"""
		self.create_working_dir()
		""" ATTENTION LA LIGNE QUI SUIT VA PROVOQUER UNE ERREUR SI LA NORME DE LA VITESSE DEPASSE vmax"""
		## Normalisation des couleurs
		self.color_norm = matplotlib.colors.Normalize(vmin=0.,vmax=2.1)
		self.color_norm_w = matplotlib.colors.Normalize(vmin=-13,vmax=13)
		## Enregistrement de la vorticité
		os.mkdir("./w/")
		os.mkdir("./vitesse/")

	def callback_gif_loop(self, n, t_simu, t_simu_precedemment_enregistre):
		plt.clf()
		# Enregistrement de la norme de la vitesse
		plt.imshow(np.sqrt(self.u[1:-1, 1:-1]**2+self.v[1:-1, 1:-1]**2), origin='lower', cmap='afmhot', interpolation = 'none', norm = self.color_norm)
		plt.colorbar()
		plt.title("t = {:.2f}".format(t_simu))
		plt.axis('image')
		if n<10:		
			plt.savefig("./vitesse/image000{}.jpg".format(n+1), dpi = self.dpi)
			file_vitesse=open('video_vitesse.txt', 'a')
			file_vitesse.write("file 'vitesse/image000{}.jpg'\n".format(n+1))
			file_vitesse.write("duration {}\n".format((t_simu-t_simu_precedemment_enregistre)/self.vitesse_video))
		elif n<100:
			plt.savefig("./vitesse/image00{}.jpg".format(n+1), dpi = self.dpi)
			file_vitesse=open('video_vitesse.txt', 'a')
			file_vitesse.write("file 'vitesse/image00{}.jpg'\n".format(n+1))
			file_vitesse.write("duration {}\n".format((t_simu-t_simu_precedemment_enregistre)/self.vitesse_video))
		elif n<1000:
			plt.savefig("./vitesse/image0{}.jpg".format(n+1), dpi = self.dpi)
			file_vitesse=open('video_vitesse.txt', 'a')
			file_vitesse.write("file 'vitesse/image0{}.jpg'\n".format(n+1))
			file_vitesse.write("duration {}\n".format((t_simu-t_simu_precedemment_enregistre)/self.vitesse_video))
		else:
			plt.savefig("./vitesse/image{}.jpg".format(n+1), dpi = self.dpi)
			file_vitesse=open('video_vitesse.txt', 'a')
			file_vitesse.write("file 'vitesse/image{}.jpg'\n".format(n+1))
			file_vitesse.write("duration {}\n".format((t_simu-t_simu_precedemment_enregistre)/self.vitesse_video))
				
		## Enregistrement de la vorticité
		plt.clf()
		plt.imshow(self.w[1:-1,1:-1], origin='lower', cmap='seismic', interpolation = 'none', norm = self.color_norm_w)
		plt.colorbar()
		plt.title("t = {:.2f}".format(t_simu))
		plt.axis('image')
		if n<10:		
			plt.savefig("./w/image000{}.jpg".format(n+1), dpi = self.dpi)
			file_w=open('video_w.txt', 'a')
			file_w.write("file 'w/image000{}.jpg'\n".format(n+1))
			file_w.write("duration {}\n".format((t_simu-t_simu_precedemment_enregistre)/self.vitesse_video))
		elif n<100:
			plt.savefig("./w/image00{}.jpg".format(n+1), dpi = self.dpi)
			file_w=open('video_w.txt', 'a')
			file_w.write("file 'w/image00{}.jpg'\n".format(n+1))
			file_w.write("duration {}\n".format((t_simu-t_simu_precedemment_enregistre)/self.vitesse_video))
		elif n<1000:
			plt.savefig("./w/image0{}.jpg".format(n+1), dpi = self.dpi)
			file_w=open('video_w.txt', 'a')
			file_w.write("file 'w/image0{}.jpg'\n".format(n+1))
			file_w.write("duration {}\n".format((t_simu-t_simu_precedemment_enregistre)/self.vitesse_video))
		else:
			plt.savefig("./w/image{}.jpg".format(n+1), dpi = self.dpi)
			file_w=open('video_w.txt', 'a')
			file_w.write("file 'w/image{}.jpg'\n".format(n+1))
			file_w.write("duration {}\n".format((t_simu-t_simu_precedemment_enregistre)/self.vitesse_video))
	def callback_gif_end(self):
		np.save("reprise.npy", np.array([self.Lx, self.Ly, self.Nx, self.Ny, self.Nt, self.r, self.Re, self.pas_enregistrement, self.dpi, self.vitesse_video, self.t_simu]))
	
	def callback_save_start(self):
		#Création du dossier pour l'enregistrement
		#self.create_working_dir()
		## Enregistrement de la vitesse
		os.mkdir("./u/")
		os.mkdir("./v/")
	
	def callback_save_loop(self, n):
		# Enregistrement de u
		os.chdir(self.dirname+"/u/")
		if n<10:		
			np.save("000{}.npy".format(n+1), self.u)
		elif n<100:
			np.save("00{}.npy".format(n+1), self.u)
		elif n<1000:
			np.save("0{}.npy".format(n+1), self.u)
		else:
			np.save("{}.npy".format(n+1), self.u)
		os.chdir(self.dirname)
				
		# Enregistrement de v
		os.chdir(self.dirname+"/v/")
		if n<10:
			np.save("000{}.npy".format(n+1), self.v)
		elif n<100:
			np.save("00{}.npy".format(n+1), self.v)
		elif n<1000:
			np.save("0{}.npy".format(n+1), self.v)
		else:
			np.save("{}.npy".format(n+1), self.v)
		os.chdir(self.dirname)
	
	def callback_save_end(self):
		pass
	"""
	---------------------------------------------------------------------------------
	|										|										|
	|				BOUCLE PRINCIPALE				|
	|										|										|
	---------------------------------------------------------------------------------
	"""
	
	def main_loop(self, f_callback_start = None, f_callback_loop = None, f_callback_end = None):
		"""Boucle principe ; appelle f_callback_start au début, f_callback_loop à chaque pas d'enregistrement et f_callback_end à la fin"""
		if f_callback_start == None:
			f_callback_start = self.callback_gif_start
		if f_callback_loop == None:
			f_callback_loop = self.callback_gif_loop
		if f_callback_end == None:
			f_callback_end = self.callback_gif_end
		
		self.t_simu = 0
		self.t_simu_precedemment_enregistre = 0

		t_i = time.time()
		
		f_callback_start()
		self.callback_save_start()
		## Boucle principale
		for n in range(self.Nt):
			#Calcul du nouveau dt pour respecter les conditions CFL
			self.condition_cfl()
			#Calcul de l'accélération convective
			Resu, Resv = self.calcul_a_conv()
			#Navier-Stokes
			self.ustar, self.vstar = self.navier_stokes(Resu, Resv)			
			
			
			#Conditions aux limites
			## Soufflerie numérique
			self.cl_soufflerie()
			## Bords de l'objet
			self.cl_objet()
			
			# Mise à jour des points fantomes
			self.points_fantome_vitesse(self.ustar, self.vstar)

			#Projection
			divstar = self.divergence(self.ustar, self.vstar)
			self.solve_laplacien(divstar[1:-1,1:-1])
			self.cl_phi()
			gradphi_x, gradphi_y = self.grad(self.phi) # code optimisable en réduisant le nombre de variables
			
			self.u[1:-1, 1:-1] = self.ustar[1:-1, 1:-1] - gradphi_x[1:-1, 1:-1]
			self.v[1:-1, 1:-1] = self.vstar[1:-1, 1:-1] - gradphi_y[1:-1, 1:-1]

			#Fin du calcul
			self.t_simu += self.dt
			self.points_fantome_vitesse(self.u, self.v)
	
			## Enregistrement
			if n%self.pas_enregistrement == 0:
				print(n, 'sur', self.Nt, " avec un reynolds de ", self.Re)
				f_callback_loop(n, self.t_simu, self.t_simu_precedemment_enregistre)
				self.t_simu_precedemment_enregistre = self.t_simu
				self.callback_save_loop(n)
		np.save("objet.npy", self.objet)
		self.callback_save_loop(self.Nt-1)
		f_callback_end()
		print("Simulation effectuée en {} s.".format(time.time()-t_i))


def loop_Re():
	Re_list = np.linspace(4,200,50)
	for Re in Re_list:
		simu = VonKarman(Re = Re, Nt = 2000, Nx = 1.5*300, Ny = 1.5*70, pas_enregistrement = 5, r = 0.5, vitesse_video = 5)
		simu.main_loop()
		os.chdir(simu.root_dir)

def single_Re(Re):
	expand = 1. #1.5 rapport entre x et y
	simu = VonKarman(Re = Re, Nt = 2000, Nx = 1.5*300, Ny = 1.5*70, pas_enregistrement = 5, r = 0.5, vitesse_video = 5)
	#simu.main_loop(simu.callback_correlations_vitesse_start, simu.callback_correlations_vitesse_loop, simu.callback_correlations_vitesse_end)
	simu.main_loop()
single_Re(500)
