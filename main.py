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






class fonction_courant():

	def __init__(self, Lx = 1, Ly = 1, Nx = 1, Ny=1, Re = 1,t_simu=1,dpi=70 , u=np.array([[1]]),v=np.array([[0]]), objet=np.array([[0]])):
		## Definition des constantes
		self.Lx = Lx
		self.Ly = Ly
		self.Re = Re
		self.t_simu=t_simu
		self.dpi=dpi
		self.objet=objet
		# Taille des tableaux
		self.Nx = int(Nx)
		self.Ny = int(Ny)

		# Taille du domaine réel
		self.nx = self.Nx-2 # 2 points fantômes
		self.ny = self.Ny-2
		
		# Pas
		self.dx = self.Lx/(self.nx-1)
		self.dy = self.Ly/(self.ny-1)

		#vitesse
		self.u = u
		self.v = v

	def grad(self, f):
		"""Renvoie le gradient de f. Schéma centré (ordre 2). L'axe des y est orienté vers le bas."""
		grad_f_x = np.zeros((self.Ny, self.Nx))
		grad_f_y = np.zeros((self.Ny, self.Nx))

		grad_f_y[1:-1, :] = (f[2:, :] - f[:-2, :])/(2*self.dy)
		grad_f_x[:, 1:-1] = (f[:, 2:] - f[:, :-2])/(2*self.dx)
		return grad_f_x, grad_f_y

	def grad2(self, f):
		"""Renvoie le gradient de f. Schéma centré (ordre 2). L'axe des y est orienté vers le bas."""
		grad_f_x = np.zeros((self.ny, self.nx))
		grad_f_y = np.zeros((self.ny, self.nx))

		grad_f_y[1:-1, :] = (f[2:, :] - f[:-2, :])/(2*self.dy)
		grad_f_x[:, 1:-1] = (f[:, 2:] - f[:, :-2])/(2*self.dx)
		return grad_f_x, grad_f_y

	def rotationnel(self):
		"""Renvoie la norme du vecteur vorticité selon e_z rentrant."""
		u_x, u_y = self.grad(self.u)
		v_x, v_y = self.grad(self.v)
		self.w = v_x-u_y
		return self.w

	def laplacien(self, f):
		"""Renvoie le laplacien de la fonction scalaire f"""
		laplacien_f = np.zeros((self.ny, self.nx))
		dx_2 = 1/(self.dx)**2
		dy_2 = 1/(self.dy)**2
		coef0 = -2*(dx_2 + dy_2)  
		laplacien_f[1:-1,1:-1] = dy_2*(f[2:,1:-1]+f[:-2,1:-1])+dx_2*(f[1:-1, 2:]+f[1:-1,:-2])+coef0*f[1:-1,1:-1]
		return laplacien_f

	def construction_matrice_laplacien_2D(self):
		"""Construit et renvoie la matrice sparse du laplacien 2D pour psi"""
		dx_2 = 1/(self.dx)**2
		dy_2 = 1/(self.dy)**2
		offsets = np.array([-1,0,1])                    

		# Axe x
		datax = [np.ones(self.nx), -2*np.ones(self.nx), np.ones(self.nx)]
		DXX = sp.dia_matrix((datax,offsets), shape=(self.nx,self.nx)) * dx_2
		DXX2 = DXX.todense()
		## Conditions aux limites : Neumann
		DXX2[0,1]     = 2.*dx_2  # SF left : au lieu d'un 1 on met un 2 à la deuxième colonne première ligne
		DXX2[-1,-2] = 2*dx_2  # SF right 

		# Axe Y
		datay = [np.ones(self.ny), -2*np.ones(self.ny), np.ones(self.ny)] 
		DYY = sp.dia_matrix((datay,offsets), shape=(self.ny,self.ny)) * dy_2	
		DYY2 = DYY.todense()  
		## Conditions aux limites : Neumann 
		DYY2[0,1]     = 2.*dy_2  # en haut
		DYY2[-1, self.ny-2] = 2.*dy_2  # en bas
		
		
		self.matrice_laplacien_2D = sp.kron(sp.eye(self.ny, self.ny), DXX2) + sp.kron(DYY2, sp.eye(self.nx, self.nx)) #sparse
		#Cependant, avec que des conditions de Neumann, il faut fixer la valeur en un point. On la fixe égale à zéro en haut à droite.
		self.matrice_laplacien_2D[0,0]=0
	
	def mise_a_jour_w(self):
		"""il faut mettre à jour le vecteur rotationnel pour prendre en compte les CL de psi."""
		#A gauche, on a v=0 donc pas besoin de le mettre à jour

		#A droite, on a v non nul à priori donc, au niveaudes points REELS de w cela donne
		for i in range(self.Ny):
			self.w[i,-2] += 2*self.v[i,-1]/self.dx
	
		#En haut, u est non nul également aussi
		for j in range(self.Nx):
			self.w[1,j] += -2*self.u[1,j]/self.dy

		#Enfin, u est non nul en bas donc
		for j in range(self.Nx):
			self.w[-2,j] += 2*self.u[-2,j]/self.dy

	def solve_laplacien(self, div):
		"""Renvoie psi telle que laplacien(psi) = div, avec les conditions aux limites données par cl_phi (cl_phi travaille directement sur les tableaux). La taille de div est nx*ny, celle de lapacien son carré"""
		div_flat = div.flatten()
		psi_flat = linalg.spsolve(self.matrice_laplacien_2D, div_flat)
		self.psi = np.zeros((self.ny,self.nx))
		self.psi = psi_flat.reshape((self.ny,self.nx))		
	



	def main(self):
		self.construction_matrice_laplacien_2D()
		self.w = self.rotationnel()
		self.mise_a_jour_w()
		#self.w=self.w*self.objet
		self.solve_laplacien(-self.w[1:-1,1:-1])	
		plt.clf()
		maxi=np.max(self.psi)
		mini=np.min(self.psi)
		#self.psi=self.psi - np.ones((self.ny,self.nx))*(maxi+mini)/2
		plt.imshow(self.psi, origin='lower',cmap='seismic')
		plt.colorbar()
		plt.imshow(self.objet,alpha=0.1, cmap='gray')
		plt.title("Re = {}".format(self.Re))
		plt.axis('image')
		plt.savefig("Fonctions_courant/Re={}_t={}.jpg".format(self.Re, self.t_simu), dpi= self.dpi)
		plt.clf()


total=['Re = 20', 'Re = 23', 'Re = 26']



for i in total:
	dirname = os.path.dirname(os.path.realpath(__file__))+"/"+i
	[Lx, Ly, Nx, Ny, N_ini, r, Re, pas_enregistrement, dpi, vitesse_video, t_simu] = list(np.load(dirname+"/"+"reprise.npy"))
	N_ini=int(N_ini)
	objet = np.load(dirname+"/objet.npy")
	if N_ini<10:		
		u=np.load(dirname+"/u/000{}.npy".format(N_ini))
	 	v=np.load(dirname+"/v/000{}.npy".format(N_ini))
	elif N_ini<100:
		u=np.load(dirname+"/u/00{}.npy".format(N_ini))
		v=np.load(dirname+"/v/00{}.npy".format(N_ini))
	elif N_ini<1000:
		u=np.load(dirname+"/u/0{}.npy".format(N_ini))
		v=np.load(dirname+"/v/0{}.npy".format(N_ini))
	else:
		u=np.load(dirname+"/u/{}.npy".format(N_ini))
		v=np.load(dirname+"/v/{}.npy".format(N_ini))
	simu = fonction_courant(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, Re=Re,t_simu=t_simu, dpi=dpi, u=u, v=v, objet=objet)
	simu.main()


"""


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap='afmhot',
                       linewidth=0)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, cmap='afmhot')
plt.clf
plt.show()
plt.savefig('image.jpg')

"""

"""

def ligne_courant(u,v, x_ini,dx):
	X=[x_ini]
	for k in range(n):
		i = X[-1][0]
		j = X[-1][1]
		if i < len(u)-5 and j< len(u[0])-3:		
			i = int(X[-1][0])
			j = int(X[-1][1])
			deltai = X[-1][0]-i
			deltaj = X[-1][1]-j
			vitesse_u=((1-deltai)*(1-deltaj)*u[i,j]+(1-deltaj)*deltai*u[i+1,j]+deltaj*(1-deltai)*u[i,j+1]+deltai*deltaj*u[i+1,j+1])
			vitesse_v=((1-deltai)*(1-deltaj)*v[i,j]+(1-deltaj)*deltai*v[i+1,j]+deltaj*(1-deltai)*v[i,j+1]+deltai*deltaj*v[i+1,j+1])
			
			print(k, x_ini)
			if vitesse_u > vitesse_v:	
				X.append([X[-1][0]+np.sign(vitesse_v)*dx*abs(vitesse_v/vitesse_u),X[-1][1]+np.sign(vitesse_u)*dx])
			else:
				X.append([X[-1][0]+dx*np.sign(vitesse_v),X[-1][1]+np.sign(vitesse_u)*dx*abs(vitesse_u/vitesse_v)])
	return(X)

sauvegarde = "image_Re=028_2.jpg"
dx=0.3
n = 12000
#a=[3, 9, 13, 19, 27, 32, 38, 43, 47, 50, 54, 59, 62, 86]
a=[   int(Ny/2-1), int(Ny/2+1), int(Ny/2+8), int(Ny/2-8),int(Ny/3), int(2*Ny/3)]

image = np.copy(objet)
for m in a :
	X = ligne_courant(u,v, [m,130],dx = 0.2)
	for k in range(len(X)):
		i = int(X[k][0])
		j = int(X[k][1])
		image[i,j] = -1
plt.imshow(image)
plt.savefig(os.path.dirname(os.path.realpath(__file__))+"/images/"+sauvegarde)
"""


