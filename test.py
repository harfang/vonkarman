import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

NXi = 6
NYi = 6

dataNXi = [np.ones(NXi), -2*np.ones(NXi), np.ones(NXi)]   
    
### Conditions aux limites : Neumann 

print(dataNXi)
dataNXi[2][1]     = 2.  # SF left
dataNXi[0][NXi-2] = 2.  # SF right

###### AXE Y
## Diagonal terms
dataNYi = [np.ones(NYi), -2*np.ones(NYi), np.ones(NYi)] 
  
### Conditions aux limites : Neumann 
dataNYi[2][1]     = 2.  # SF low
dataNYi[0][NYi-2] = 2.  # SF top

###### Their positions
offsets = np.array([-1,0,1])                    
DXX = sp.dia_matrix((dataNXi,offsets), shape=(NXi,NXi))
DYY = sp.dia_matrix((dataNYi,offsets), shape=(NYi,NYi))

DXX2 = DXX.todense()
DYY2 = DYY.todense()

DXX2[0,:] = np.zeros(DXX2[0,:].shape)
DXX2[-1,:] = np.zeros(DXX2[-1,:].shape)

DYY2[0,:] = np.zeros(DYY2[0,:].shape)
DYY2[-1,:] = np.zeros(DYY2[-1,:].shape)

DX = sp.dia_matrix.todia(DXX2)
DY = sp.dia_matrix.todia(DYY2)

LAP = sp.kron(sp.eye(NYi,NYi), DX) + sp.kron(DY, sp.eye(NXi,NXi))

print(DX)
print(DY)
print(LAP)
