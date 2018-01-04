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

def harmonics(freqs, spectre, cutoff_freq_low = None, cutoff_freq_high = None, return_indices = False, largeur = 100):
	"""Renvoie la liste des maximums locaux du spectre (fréquences et amplitude), ayant une fréquence entre cutoff_freq_low et cutoff_freq_high
	
	largueur : largueur de chaque pic en Hz (un pic à la fréquence f est un maximum local s'il est le maximum sur l'intervalle f +- largeur"""
	
	cutoff_low = 0
	if cutoff_freq_low is not None:
		cutoff_low = np.where(freqs >= cutoff_freq_low)[0][0]
	cutoff_high = len(freqs)-1
	if cutoff_freq_high is not None:
		cutoff_high = np.where(freqs <= cutoff_freq_high)[0][-1]
	order = int(largeur/abs(freqs[1]-freqs[0]))
	indices = scipy.signal.argrelextrema(spectre[cutoff_low:cutoff_high], np.greater, order=order)[0]
	if return_indices:
		indices += cutoff_low
		return freqs[cutoff_low:cutoff_high][indices], spectre[cutoff_low:cutoff_high][indices], indices
	else:
		return freqs[cutoff_low:cutoff_high][indices], spectre[cutoff_low:cutoff_high][indices]

"""
---------------------------------------------------------------------------------
|																				|
|				CONSTANTES ET OBJETS											|
|																				|
---------------------------------------------------------------------------------
"""

class VonKarman():

	def __init__(self, Lx = 30.0, Ly = 10.0, Nx = 150, Ny=50, Nt=1200, r = 0.35, Re = 100, pas_enregistrement = 5, dpi = 100, vitesse_video = 5, colorant_initial= []):
		## Definition des constantes
		self.Lx = Lx
		self.Ly = Ly
		
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

		# Conditions initiales
		self.u = np.zeros((self.Ny, self.Nx))
		self.v = np.zeros((self.Ny, self.Nx))
		self.phi = np.zeros((self.Ny, self.Nx))
		self.ustar = np.zeros((self.Ny, self.Nx))
		self.vstar = np.zeros((self.Ny, self.Nx))
		self.colorant=[]
		self.colorant_initial =	colorant_initial
		
		# Définition de l'objet
		#On place le centre de l'objet en -8r, Ly/2)
		#la matrice objet renvoie une matrice pleine de 0 là où il y a l'objet et pleine de 1 là où il n'y est pas
		self.objet=np.ones((self.Ny, self.Nx))
		for i in range(self.Ny): 
			for j in range(self.Nx):
#				if (j*self.dx-15*self.r)**2+(i*self.dy-0.3*(self.Ly+2*self.dy))**2 < self.r**2:
#					self.objet[i][j]=0
#				if (j*self.dx-15*self.r)**2+(i*self.dy-0.6*(self.Ly+2*self.dy))**2 < self.r**2:
#					self.objet[i][j]=0
				if (j*self.dx-15*self.r)**2+(i*self.dy-0.51*(self.Ly+2*self.dy))**2 < self.r**2:
					self.objet[i][j]=0
		
		self.grille_colorant = np.copy(self.objet)
		
		# Affichage
		self.dirname = ""
		self.root_dir = ""
		self.dpi = int(dpi)
		for k in self.colorant_initial:
			self.colorant.append([k[0],k[1]])		
		
		# Laplacien
		self.construction_matrice_laplacien_2D()
	
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
	|																				|
	|				AUTRES OBSERVABLES												|
	|																				|
	---------------------------------------------------------------------------------
	"""
	
	@property
	def w(self):
		"""Renvoie la norme du vecteur vorticité"""
		u_x, u_y = self.grad(self.u)
		v_x, v_y = self.grad(self.v)
		
		return v_x-u_y
		
	def colo(self):
		"""Cette fonction met à jour la position de chaque particule de colorant en calculant sa vitesse et donc sa position en t+dt. Colorant est une liste attention !! """
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
		dt_diffusion = facteur_de_precaution_cfl * self.Re*min(self.dx**2, self.dy**2)/10 #/4 au lieu de 10 ; /5 : trop peu pour Re ~ 60, /10 : mieux mais à tester
	
		#3. min
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
	|																				|
	|				LES GROSSES FONCTIONS											|
	|																				|
	---------------------------------------------------------------------------------
	"""

	def calcul_a_conv(self):
		"""Calcul l'accélération convective u grad u"""
		#problème : il utilise une autre convention que la notre c'est-à-dire que pour lui x est le premier indice. 

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
		DXX2[-1,-1] = -3*dx_2  # SF right 0 ou -1 ???
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

	"""
	---------------------------------------------------------------------------------
	|																				|
	|				AFFICHAGE ET ENREGISTREMENT										|
	|																				|
	---------------------------------------------------------------------------------
	"""
	
	def create_working_dir(self):
		self.root_dir = os.path.dirname(os.path.realpath(__file__))
		#self.dirname = self.root_dir+"\\"+str(time.time())+" - Re = "+str(self.Re)
		self.dirname = self.root_dir+"\\Re = "+str(self.Re)
		if not os.path.exists(self.dirname):
			os.makedirs(self.dirname)
		else:
			timestamp = time.time()
			self.dirname += str(timestamp-int(timestamp))[1:]
			os.makedirs(self.dirname)
	
		os.chdir(self.dirname) # change le répertoire de travail

		np.savetxt("parameters.txt", np.array([self.Lx, self.Ly, self.Nx, self.Ny, self.Nt, self.r, self.Re, self.Nt, self.pas_enregistrement]).transpose(), header = "Lx \t Ly \t Nx \t Ny \t Nt \t r \t Re \t Nt \t pas_enregistrement \n", newline="\t")
	
	""" ENREGISTREMENT DE GIF ANIME """
	
	def callback_video_start(self):
		"""Création du dossier pour l'enregistrement"""
		self.create_working_dir()
	
		## Préparation de la création d'un gif animé
		self.command_string_video = ""
		self.command_string_video_w = ""

		""" ATTENTION LA LIGNE QUI SUIT VA PROVOQUER UNE ERREUR SI LA NORME DE LA VITESSE DEPASSE vmax"""
		## Normalisation des couleurs
		self.color_norm = matplotlib.colors.Normalize(vmin = 0.,vmax = 2.1)
		self.color_norm_w = matplotlib.colors.Normalize(vmin = -13,vmax = 13)
		self.color_norm_colorant = matplotlib.colors.Normalize(vmin = -1,vmax = 5)
		
		## Enregistrement de la vorticité
		os.mkdir("./w/")
		os.mkdir("./vitesse/")
		os.mkdir("./colorant/")
		
	def callback_video_loop(self, n, t_simu, t_simu_precedemment_enregistre):
		plt.clf()
		# Enregistrement de la norme de la vitesse
		plt.imshow(np.sqrt(self.u[1:-1, 1:-1]**2+self.v[1:-1, 1:-1]**2), origin='lower', cmap='gray', interpolation = 'none', norm = self.color_norm)
		plt.colorbar()
		plt.title("t = {:.2f}".format(t_simu))
		plt.axis('image')
		if n<10:		
			plt.savefig("./vitesse/image000{}.jpg".format(n+1), dpi = self.dpi)
			self.command_string_video += "file 'vitesse/image000{}.jpg'\n".format(n+1)
		elif n<100:
			plt.savefig("./vitesse/image00{}.jpg".format(n+1), dpi = self.dpi)
			self.command_string_video += "file 'vitesse/image00{}.jpg'\n".format(n+1)
		elif n<1000:
			plt.savefig("./vitesse/image0{}.jpg".format(n+1), dpi = self.dpi)
			self.command_string_video += "file 'vitesse/image0{}.jpg'\n".format(n+1)
		else:
			plt.savefig("./vitesse/image{}.jpg".format(n+1), dpi = self.dpi)
			self.command_string_video += "file 'vitesse/image{}.jpg'\n".format(n+1)
		
		self.command_string_video += "duration {}\n".format((t_simu-t_simu_precedemment_enregistre)/self.vitesse_video)
		
		## Enregistrement de la vorticité
		plt.clf()
		plt.imshow(self.w[1:-1,1:-1], origin='lower', cmap='seismic', interpolation = 'none', norm = self.color_norm_w)
		plt.colorbar()
		plt.title("t = {:.2f}".format(t_simu))
		plt.axis('image')
		if n<10:		
			plt.savefig("./w/image000{}.jpg".format(n+1), dpi = self.dpi)
			self.command_string_video_w += "file 'w/image000{}.jpg'\n".format(n+1)
		elif n<100:
			plt.savefig("./w/image00{}.jpg".format(n+1), dpi = self.dpi)
			self.command_string_video_w += "file 'w/image00{}.jpg'\n".format(n+1)
		elif n<1000:
			plt.savefig("./w/image0{}.jpg".format(n+1), dpi = self.dpi)
			self.command_string_video_w += "file 'w/image0{}.jpg'\n".format(n+1)
		else:
			plt.savefig("./w/image{}.jpg".format(n+1), dpi = self.dpi)
			self.command_string_video_w += "file 'w/image{}.jpg'\n".format(n+1)
		
		self.command_string_video_w += "duration {}\n".format((t_simu-t_simu_precedemment_enregistre)/self.vitesse_video)		
		
#		## Enregistrement du colorant
#		plt.clf()
#		self.colorant=self.colo()
#		self.grille_colorant=np.copy(self.objet)
#		for k in range(len(self.colorant)):
#			if self.grille_colorant[int(round(self.colorant[k][0]))][int(round(self.colorant[k][1]))] < 6:
#				self.grille_colorant[int(round(self.colorant[k][0]))][int(round(self.colorant[k][1]))]+=1
#		plt.imshow(self.grille_colorant, origin='lower',  cmap='afmhot', interpolation = 'none')		
#		plt.colorbar()
#		plt.title("t = {:.2f}".format(t_simu))
#		plt.axis('image')
#		if n<10:		
#			plt.savefig("./colorant/image000{}.jpg".format(n+1), dpi = self.dpi)
#			file_colorant=open('video_colorant.txt', 'a')
#			file_colorant.write("file 'colorant/image000{}.jpg'\n".format(n+1))
#			file_colorant.write("duration {}\n".format((t_simu-t_simu_precedemment_enregistre)/self.vitesse_video))
#			print(n)
#		elif n<100:
#			plt.savefig("./colorant/image00{}.jpg".format(n+1), dpi = self.dpi)
#			file_colorant=open('video_colorant.txt', 'a')
#			file_colorant.write("file 'colorant/image00{}.jpg'\n".format(n+1))
#			file_colorant.write("duration {}\n".format((t_simu-t_simu_precedemment_enregistre)/self.vitesse_video))
#		elif n<1000:
#			plt.savefig("./colorant/image0{}.jpg".format(n+1), dpi = self.dpi)
#			file_colorant=open('video_colorant.txt', 'a')
#			file_colorant.write("file 'colorant/image0{}.jpg'\n".format(n+1))
#			file_colorant.write("duration {}\n".format((t_simu-t_simu_precedemment_enregistre)/self.vitesse_video))
#		else:
#			plt.savefig("./colorant/image{}.jpg".format(n+1), dpi = self.dpi)
#			file_colorant=open('video_colorant.txt', 'a')
#			file_colorant.write("file 'colorant/image{}.jpg'\n".format(n+1))
#			file_colorant.write("duration {}\n".format((t_simu-t_simu_precedemment_enregistre)/self.vitesse_video))

	def callback_video_end(self):
		with open("video_vitesse.txt", "w") as saved_commandstring_video:
			saved_commandstring_video.write(self.command_string_video)
		with open("video_w.txt", "w") as saved_commandstring_video_w:
			saved_commandstring_video_w.write(self.command_string_video_w)
#		with open("video_colorant.txt", "w") as saved_commandstring_video_colorant:
#			saved_commandstring_video.write(self.command_string_video_colorant)
		os.chdir(self.dirname)
		#os.system("ffmpeg -f concat -i video_vitesse.txt vitesse.mp4")
		#os.system("ffmpeg -f concat -i video_w.txt vorticite.mp4")
		print("Données enregistrées dans {}.".format(self.dirname))
	
	""" ANCIENNE VERSION : FABRICATION DE GIFS """
	
	def callback_gif_start(self):
		"""Création du dossier pour l'enregistrement"""
		self.create_working_dir()
	
		## Préparation de la création d'un gif animé
		self.command_string = "convert *.jpg -loop 0 \\\n"

		""" ATTENTION LA LIGNE QUI SUIT VA PROVOQUER UNE ERREUR SI LA NORME DE LA VITESSE DEPASSE vmax"""
		## Normalisation des couleurs
		self.color_norm = matplotlib.colors.Normalize(vmin=0.,vmax=2.1)
		self.color_norm_w = matplotlib.colors.Normalize(vmin=-13,vmax=13)
		
		## Enregistrement de la vorticité
		os.mkdir("./w/")
	
	def callback_gif_loop(self, n, t_simu, t_simu_precedemment_enregistre):
		plt.clf()
		# Enregistrement de la norme de la vitesse
		plt.imshow(np.sqrt(self.u[1:-1, 1:-1]**2+self.v[1:-1, 1:-1]**2), origin='lower', cmap='gray', interpolation = 'none', norm = self.color_norm)
		plt.colorbar()
		plt.title("t = {:.2f}".format(t_simu))
		plt.axis('image')
		if n<10:		
			plt.savefig("image000{}_t={}_Re={}.jpg".format(n+1, t_simu, self.Re), dpi = self.dpi)
		elif n<100:
			plt.savefig("image00{}_t={}_Re={}.jpg".format(n+1, t_simu, self.Re), dpi = self.dpi)
		elif n<1000:
			plt.savefig("image0{}_t={}_Re={}.jpg".format(n+1, t_simu, self.Re), dpi = self.dpi)
		else:
			plt.savefig("image{}_t={}_Re={}.jpg".format(n+1, t_simu, self.Re), dpi = self.dpi)
				
		## Enregistrement de la vorticité
		plt.clf()
		plt.imshow(self.w[1:-1,1:-1], origin='lower', cmap='seismic', interpolation = 'none', norm = self.color_norm_w)
		plt.colorbar()
		plt.title("t = {:.2f}".format(t_simu))
		plt.axis('image')
		if n<10:		
			plt.savefig("./w/image000{}_t={}_Re={}.jpg".format(n+1, t_simu, self.Re), dpi = self.dpi)
		elif n<100:
			plt.savefig("./w/image00{}_t={}_Re={}.jpg".format(n+1, t_simu, self.Re), dpi = self.dpi)
		elif n<1000:
			plt.savefig("./w/image0{}_t={}_Re={}.jpg".format(n+1, t_simu, self.Re), dpi = self.dpi)
		else:
			plt.savefig("./w/image{}_t={}_Re={}.jpg".format(n+1, t_simu, self.Re), dpi = self.dpi)

		self.command_string += "\\( -clone {} -set delay {} \\) -swap {} +delete \\\n".format(n//self.pas_enregistrement, (t_simu-t_simu_precedemment_enregistre)*10, n//self.pas_enregistrement)
	
	def callback_gif_end(self):
		self.command_string += " movie.gif"
		os.system(self.command_string)
		os.chdir(self.dirname+"/w/")
		os.system(self.command_string)
		os.chdir(self.dirname)
		with open("commandstring.txt", "w") as saved_commandstring:
			saved_commandstring.write(self.command_string)
		print("Données enregistrées dans {}.".format(self.dirname))
		
	""" FIN ANCIENNE VERSION """
	
	""" CORRELATIONS DE VITESSE"""
	
	def callback_correlations_vitesse_start(self):
		self.create_working_dir()
		self.amplitude_oscillations = np.zeros((self.Nt//self.pas_enregistrement))
		self.liste_t = np.zeros((self.Nt//self.pas_enregistrement))
		
#		ny_test = int(self.r/self.dy)+5
#		nx_test = int(self.r/self.dx)+5
#		self.objet_test = np.ones((ny_test, nx_test))
#		for i in range(ny_test):
#			for j in range(nx_test):
#				if (j*self.dx-(nx_test*self.dx)/2)**2+(i*self.dy-(ny_test*self.dy)/2)**2 < self.r**2:
#					self.objet_test[i][j] = 0.
					
		self.index_bord_calcul_correlations = np.argmin(self.objet, axis = 1).max()+int(6*self.r/self.dy)
		
	def callback_correlations_vitesse_loop(self, n, t_simu, t_simu_precedemment_enregistre):
#		norme_vitesse = np.sqrt(self.u[1:-1,self.index_bord_calcul_correlations:-1]**2+self.v[1:-1,self.index_bord_calcul_correlations:-1]**2)
#		corr = scipy.signal.correlate2d(norme_vitesse, self.objet_test, mode = "same", boundary = "symm")
#		self.amplitude_oscillations[n//self.pas_enregistrement] = float(np.argmin(corr, axis = 0)[0])
#		self.liste_t[n//self.pas_enregistrement] = t_simu
		
		#norme_vitesse = np.sqrt(self.u[1:-1,self.index_bord_calcul_correlations]**2+self.v[1:-1,self.index_bord_calcul_correlations]**2)
		self.amplitude_oscillations[n//self.pas_enregistrement] = float(np.argmin(self.w[1:-1,self.index_bord_calcul_correlations]))
		self.liste_t[n//self.pas_enregistrement] = t_simu
		
		"""
		plt.clf()
		plt.imshow(corr, origin='lower', cmap='bwr', interpolation = 'none')
		plt.colorbar()
		plt.title("t = {:.2f}".format(t_simu))
		plt.axis('image')
		if n<10:		
			plt.savefig("image000{}_t={}_Re={}.jpg".format(n+1, t_simu, self.Re), dpi = self.dpi)
		elif n<100:
			plt.savefig("image00{}_t={}_Re={}.jpg".format(n+1, t_simu, self.Re), dpi = self.dpi)
		elif n<1000:
			plt.savefig("image0{}_t={}_Re={}.jpg".format(n+1, t_simu, self.Re), dpi = self.dpi)
		else:
			plt.savefig("image{}_t={}_Re={}.jpg".format(n+1, t_simu, self.Re), dpi = self.dpi)
		"""
		
	def callback_correlations_vitesse_end(self):
		np.savetxt("amplitude_oscillations.txt", np.array(self.amplitude_oscillations))
		np.savetxt("liste_t.txt", np.array(self.liste_t))
		plt.clf()
		plt.plot(self.liste_t[1000:], self.amplitude_oscillations[1000:])
		plt.savefig("plot_amplitude_Re={}.jpg".format(self.Re), dpi = 300)
		plt.clf()
		freqs = np.linspace(1, 70, 40000)
		pgram = scipy.signal.lombscargle(np.array(self.liste_t, dtype = 'float64')[1000:], np.array(self.amplitude_oscillations, dtype = 'float64')[1000:], freqs)/(self.Nt//self.pas_enregistrement)
		indices = scipy.signal.argrelextrema(pgram, np.greater, order = 2000)
		freqs_max = freqs[indices]
		print(freqs_max)

		plt.plot(freqs, pgram)
		plt.plot(freqs[indices], pgram[indices], linestyle ='', marker = '+')
		plt.savefig("plot_amplitude_spectre_Re={}.jpg".format(self.Re), dpi = 300)
		np.savetxt("freqs_max.txt", freqs_max)
		os.chdir(self.dirname)
		with open("../dataStRe.txt", "a") as f:
			f.write("{},{}\n".format(self.Re, freqs_max*self.r))
		
	""" ENREGISTREMENT DES DONNEES """
	
	def callback_save_start(self):
		#Création du dossier pour l'enregistrement
		self.create_working_dir()
		
		## Enregistrement de la vitesse
		os.mkdir("./u/")
		os.mkdir("./v/")
	
	def callback_save_loop(self, n, t_simu, t_simu_precedemment_enregistre):
		# Enregistrement de u
		os.chdir(self.dirname+"/u/")
		if n<10:		
			np.save("000{}_t={}_Re={}.npy".format(n+1, t_simu, self.Re), self.u)
		elif n<100:
			np.save("00{}_t={}_Re={}.npy".format(n+1, t_simu, self.Re), self.u)
		elif n<1000:
			np.save("0{}_t={}_Re={}.npy".format(n+1, t_simu, self.Re), self.u)
		else:
			np.save("{}_t={}_Re={}.npy".format(n+1, t_simu, self.Re), self.u)
		os.chdir(self.dirname)
				
		# Enregistrement de v
		os.chdir(self.dirname+"/v/")
		if n<10:
			np.save("000{}_t={}_Re={}.npy".format(n+1, t_simu, self.Re), self.v)
		elif n<100:
			np.save("00{}_t={}_Re={}.npy".format(n+1, t_simu, self.Re), self.v)
		elif n<1000:
			np.save("0{}_t={}_Re={}.npy".format(n+1, t_simu, self.Re), self.v)
		else:
			np.save("{}_t={}_Re={}.npy".format(n+1, t_simu, self.Re), self.v)
		os.chdir(self.dirname)
	
	def callback_save_end(self):
		pass
		
	"""
	---------------------------------------------------------------------------------
	|																				|
	|				BOUCLE PRINCIPALE												|
	|																				|
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
		
		t_simu = 0
		t_simu_precedemment_enregistre = 0

		t_i = time.time()
		
		f_callback_start()

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
			t_simu += self.dt
			self.points_fantome_vitesse(self.u, self.v)
	
			## Enregistrement
			if n%self.pas_enregistrement == 0:
				#print(n, 'sur', self.Nt)
				f_callback_loop(n, t_simu, t_simu_precedemment_enregistre)
				t_simu_precedemment_enregistre = t_simu
				
		f_callback_end()
		print("Simulation effectuée en {} s.".format(time.time()-t_i))


def loop_Re():
	invRe = np.linspace(1/1e15, 1/150, 50)
	#Re_list = np.logspace(2.6, 5, 5)
	Re_list = 1/invRe
#	Re_list = np.array([1e10, 1e9])
	expand = 1.3
	for Re in Re_list:
		simu = VonKarman(Re = Re, Nt = 6000, Nx = expand*300, Ny = expand*100, pas_enregistrement = 1, r = 0.5)
		#simu.main_loop(simu.callback_video_start, simu.callback_video_loop, simu.callback_video_end)
		simu.main_loop(simu.callback_correlations_vitesse_start, simu.callback_correlations_vitesse_loop, simu.callback_correlations_vitesse_end)
		os.chdir(simu.root_dir)

def single_Re(Re):
	expand = 1.5 #1.5
	simu = VonKarman(Re = Re, Nt = 2000, Nx = expand*300, Ny = expand*70, pas_enregistrement = 5, r = 0.5, dpi = 160)
	simu.main_loop(simu.callback_correlations_vitesse_start, simu.callback_correlations_vitesse_loop, simu.callback_correlations_vitesse_end)
	#simu.main_loop()
	#simu.main_loop(simu.callback_video_start, simu.callback_video_loop, simu.callback_video_end)
#single_Re(100)
loop_Re()
#data = np.loadtxt("dataStRe.txt", delimiter = ',')
#plt.plot(1/data[:, 0], data[:, 1])
