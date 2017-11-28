import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

#NXi = 2
#NYi = 5

#dataNXi = [np.ones(NXi), -2*np.ones(NXi), np.ones(NXi)]   
#    
#### Conditions aux limites : Neumann 

#print(dataNXi)
#dataNXi[2][1]     = 2.  # SF left
#dataNXi[0][NXi-2] = 2.  # SF right

####### AXE Y
### Diagonal terms
#dataNYi = [np.ones(NYi), -2*np.ones(NYi), np.ones(NYi)] 
#  
#### Conditions aux limites : Neumann 
#dataNYi[2][1]     = 2.  # SF low
#dataNYi[0][NYi-2] = 2.  # SF top

####### Their positions
#offsets = np.array([-1,0,1])                    
#DXX = sp.dia_matrix((dataNXi,offsets), shape=(NXi,NXi))
#DYY = sp.dia_matrix((dataNYi,offsets), shape=(NYi,NYi))

#print(sp.kron(sp.eye(NYi,NYi), DXX) + sp.kron(DYY, sp.eye(NXi,NXi)).todense())

#DXX2 = DXX.todense()
#DYY2 = DYY.todense()

#DXX2[0,:] = np.zeros(DXX2[0,:].shape)
#DXX2[-1,:] = np.zeros(DXX2[-1,:].shape)

#DYY2[0,:] = np.zeros(DYY2[0,:].shape)
#DYY2[-1,:] = np.zeros(DYY2[-1,:].shape)

#DX = sp.dia_matrix.todia(DXX2)
#DY = sp.dia_matrix.todia(DYY2)

#LAP = sp.kron(sp.eye(NYi,NYi), DX) + sp.kron(DY, sp.eye(NXi,NXi))

#print(DX)
#print(DY)
#print(LAP)

dx = 1
dy = 1

Nx = 10
Ny = 30

def construction_matrice_laplacien_2D(Nx, Ny):
	"""Construit et renvoie la matrice sparse du laplacien 2D"""
	dx_2 = 1/(dx)**2
	dy_2 = 1/(dy)**2
	# Axe y
	datax = [np.ones(Nx), -2*np.ones(Nx), np.ones(Nx)]
		
#	## Conditions aux limites : Neumann 
#	datax[2][1]     = 2.  # SF left
#	datax[0][Nx-2] = 2.  # SF right

#	# Axe Y
	datay = [np.ones(Ny), -2*np.ones(Ny), np.ones(Ny)] 
#	  
#	## Conditions aux limites : Neumann 
#	datay[2][1]     = 2.  # SF low
#	datay[0][Ny-2] = 2.  # SF top

	# Construction de la matrice sparse
	offsets = np.array([-1,0,1])               
	DXX = sp.dia_matrix((datax,offsets), shape=(Nx,Nx)) * dx_2
	DYY = sp.dia_matrix((datay,offsets), shape=(Ny,Ny)) * dy_2
	
	lap2D = sp.kron(DXX, sp.eye(Ny,Ny)) + sp.kron(sp.eye(Nx,Nx), DYY) #sparse
	
	dense_lap = lap2D.todense()
	
	# CL
	for j in range(Ny):
		# CL en haut
		dense_lap[j, :] = np.zeros((1, Nx*Ny))
		dense_lap[j,j] = 1
		dense_lap[j, 2*Ny+j] = -1
		
		# Cl en bas
		dense_lap[j+Ny*(Nx-1), :] = np.zeros((1, Nx*Ny))
		dense_lap[j+Ny*(Nx-1), j+Ny*(Nx-1)] = 1
		dense_lap[j+Ny*(Nx-1), j+Ny*(Nx-3)] = -1
		
	for i in range(1, Nx-1):
		# CL à gauche
		dense_lap[Ny*i, :] = np.zeros((1, Nx*Ny))
		dense_lap[Ny*i, Ny*i] = 1
		dense_lap[Ny*i, Ny*i+2] = -1
	 
		# CL à droite
		dense_lap[Ny*i+Ny-1, :]	= np.zeros((1, Nx*Ny))
		dense_lap[Ny*i+Ny-1, Ny*i+Ny-1] = 1
		
	lap2D_sparse = sp.dia_matrix(dense_lap)
	
	return dense_lap
	
M = construction_matrice_laplacien_2D(Nx,Ny)

U = np.ones(Nx*Ny)
print(np.dot(M, U))

