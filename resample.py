# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 20:11:43 2017

@author: Florian
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

def find_indices_min(a):
	min_value = a.min()
	indices = []
	for i in range(len(a)):
		if a[i] == min_value:
			indices.append(i)
			
	return indices
	

def calcul_tf(dirname = 'datafreq_Ly=10-7-Nt=8000'):

	dir_list = os.listdir('./{}/'.format(dirname))
	#liste_Re = np.array([float(Re[5:]) for Re in dir_list])
	liste_Re = []
	liste_Sr = []
	
	for directory in dir_list:
		a_full = np.loadtxt('{}/{}/amplitude_oscillations.txt'.format(dirname, directory))
		t_full = np.loadtxt('{}/{}/liste_t.txt'.format(dirname, directory))
		a = a_full[2000:]
		t = t_full[2000:]
	#	if a.max() != a.min():
	#		a = a/abs(a.max()-a.min())
	#	else:
	#		a = a/abs(a.max())
		#print(float(directory[5:]))
		if float(directory[5:])<65:
			pass
		else:
			liste_Re.append(float(directory[5:]))
			liste_dts = [t[i+1]-t[i] for i in range(len(t)-1)]
			dt = np.min(liste_dts)
			
			new_t = np.arange(t[0], t[-1]+dt, dt)
			new_a = np.zeros(new_t.shape)
			
			new_a[0] = a[0]
			new_a[-1] = a[-1]
			j = 0
			for i in range(1, len(t)-1):
				k = 0
				while new_t[j] <= t[i]:
					new_a[j] = a[i] + k*dt*(a[i+1]-a[i])/((t[i+1]-t[i]))
					j += 1
					k += 1
			
			#plt.plot(t, a)
			#plt.plot(new_t[0:-2], new_a[0:-2])
			freqs = fft.fftfreq(len(new_a[0:-2]), dt)
			fft_a = fft.fft(new_a[0:-2])
			#fft_a = fft_a/np.abs(fft_a[0])
			freqs_max, spectre_max = harmonics(fft.fftshift(freqs), fft.fftshift(np.abs(fft_a)), cutoff_freq_low = 0.1, cutoff_freq_high = 50, largeur = 0.04)
#			if 397<float(directory[5:]):
#				plt.plot(fft.fftshift(freqs), fft.fftshift(np.abs(fft_a)), label = directory, marker = '')
#				plt.plot(freqs_max, spectre_max, linestyle ='', marker = '+', label = directory)
#				plt.legend(loc = "best")
	#		if spectre_max[0]<1000:
	#			freqs_max[0] = 0
			#print(freqs_max)
			liste_Sr.append(freqs_max[0])
	
	liste_Re = np.array(liste_Re)
	plt.plot(1/liste_Re, liste_Sr, linestyle = '', marker = '+')
	for Re in liste_Re:
		print("{}, {}".format(Re, 1/Re))
	for Sr in liste_Sr:
		print(Sr)
		
def calcul_min(dirname = 'datafreq_Ly=10'):
	dir_list = os.listdir('./{}/'.format(dirname))
	#liste_Re = np.array([float(Re[5:]) for Re in dir_list])
	liste_Re = []
	liste_Sr = []
	incert_Sr = []
	
	for directory in dir_list:
		a_full = np.loadtxt('{}/{}/amplitude_oscillations.txt'.format(dirname, directory))
		t_full = np.loadtxt('{}/{}/liste_t.txt'.format(dirname, directory))
		a = a_full[2000:]
		t = t_full[2000:]
	#	if a.max() != a.min():
	#		a = a/abs(a.max()-a.min())
	#	else:
	#		a = a/abs(a.max())
		#print(float(directory[5:]))
		if float(directory[5:])<65:
			pass
		else:
			liste_Re.append(float(directory[5:]))
			liste_dts = [t[i+1]-t[i] for i in range(len(t)-1)]
			dt = np.mean(liste_dts)

			indices_min = find_indices_min(a)
#			plt.plot(t, a)
#			plt.plot(t[indices_min], a[indices_min], linestyle = '', marker = '+')
			liste_ecarts_t = [t[indices_min][i+1]-t[indices_min][i] for i in range(len(t[indices_min])-1)]
			liste_delta_t = []
			for k, deltat in enumerate(liste_ecarts_t):
				if deltat >= 10*dt:
					liste_delta_t.append(deltat)
			periode = np.mean(liste_delta_t)
			std_periode = np.std(liste_delta_t)
			std_Sr = std_periode/(periode**2)
			liste_Sr.append(1/periode)
			incert_Sr.append(std_Sr)
		
	liste_Re = np.array(liste_Re)
	plt.errorbar(1/liste_Re, liste_Sr, yerr = incert_Sr, linestyle = '', marker = '+')
	for Re in liste_Re:
		print("{}, {}".format(Re, 1/Re))
	for Sr in liste_Sr:
		print(Sr)
		
def calcul_lomb(dirname = 'datafreq_Ly=10-7-Nt=8000'):

	dir_list = os.listdir('./{}/'.format(dirname))
	#liste_Re = np.array([float(Re[5:]) for Re in dir_list])
	liste_Re = []
	liste_Sr = []
	
	for directory in dir_list:
		ti = time.time()
		a_full = np.loadtxt('{}/{}/amplitude_oscillations.txt'.format(dirname, directory))
		t_full = np.loadtxt('{}/{}/liste_t.txt'.format(dirname, directory))
		a = a_full[2000:]
		t = t_full[2000:]
	#	if a.max() != a.min():
	#		a = a/abs(a.max()-a.min())
	#	else:
	#		a = a/abs(a.max())
		#print(float(directory[5:]))
		if float(directory[5:])<10:
			pass
		else:
			omega = np.linspace(0.01*2*math.pi, 0.3*2*math.pi, 100000)
			pgram = scipy.signal.lombscargle(t, a, omega)
			freqs_max, spectre_max = harmonics(omega/(2*math.pi), pgram, cutoff_freq_low = 0.13, cutoff_freq_high = 0.2, largeur = 0.04)
			if len(freqs_max) > 0:
				liste_Re.append(float(directory[5:]))
				liste_Sr.append(freqs_max[0])
#			plt.plot(omega/(2*math.pi), pgram, label = directory)
#			plt.plot(freqs_max, spectre_max, linestyle = '', marker = '+')
			print("Re = {}, temps de calcul : {}".format(float(directory[5:]), time.time()-ti))
#	plt.legend(loc="best")
	liste_Re = np.array(liste_Re)
	np.savetxt(dirname+"_Re.txt", liste_Re)
	np.savetxt(dirname+"_Sr.txt", liste_Sr)
#	for Re in liste_Re:
#		print("{}, {}".format(Re, 1/Re))
#	for Sr in liste_Sr:
#		print(Sr)

def print_lomb(liste_dir, color = None, label = False):
	for dirname in liste_dir:
		liste_Re = np.loadtxt(dirname+"_Re.txt")
		liste_Sr = np.loadtxt(dirname+"_Sr.txt")
		if color is not None and label:
			plt.plot(1/liste_Re, liste_Sr, linestyle = '', marker = '+', color=color, label = dirname)
			plt.legend(loc="best")
		elif color is not None and not label:
			plt.plot(1/liste_Re, liste_Sr, linestyle = '', marker = '+', color=color)
		elif color is None and label:
			plt.plot(1/liste_Re, liste_Sr, linestyle = '', marker = '+', label = dirname)
			plt.legend(loc="best")
		else:
			plt.plot(1/liste_Re, liste_Sr, linestyle = '', marker = '+')
			
	plt.show()

def plot_th():
	a = 1.131
	Re = np.logspace(2, 11, 10000)
	plt.plot(1/(a*Re), 0.198*(1-19.7/(a*Re))/a, marker = '', color= 'k')
		
liste_dir1 = ['datafreq_Ly=10', 'datafreq_Ly=10-2', 'datafreq_Ly=10-3', 'datafreq_Ly=10-8', 'datafreq_Ly=10-9', 'datafreq_Ly=10-10']
liste_dir2 = ['datafreq_Ly=10-4', 'datafreq_Ly=10-5']
liste_dir4 = ['datafreq_Ly=10-6-Nt=4000', 'datafreq_Ly=10-7-Nt=8000']
liste_dir3 = ['datafreq_Ly=30', 'datafreq_Ly=30-2']
liste_dir5 = ['datafreq_Ly=10_expand=0.5', 'datafreq_Ly=10_expand=0.9', 'datafreq_Ly=10_expand=0.7', 'datafreq_Ly=10_expand=0.3']
#for dirname in liste_dir5:
#	calcul_lomb(dirname)
print_lomb(liste_dir1+liste_dir2, color = "red")
print_lomb(liste_dir5, label = True)
#calcul_tf("datafreq_Ly=10_expand=0.3")
#print_lomb(["datafreq_Ly=10_expand=0.3"], color = "green")
#plot_th()