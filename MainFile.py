import numpy as np
import warnings as w
from numpy import sin, cos
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import nquad
import vegas
import psutil
from multiprocess import Pool 
from scipy.interpolate import RegularGridInterpolator

np.random.seed(8869)

class Find_decent_Vrel:

	def __init__(self, pos, vel, dist):

		statement = "\nOrder of parameters: [X, Y, Z]; [Vx, Vy, Vz]; distance"
		w.warn(statement, SyntaxWarning, stacklevel = 2)

		self.x, self.y, self.z = pos
		self.vx, self.vy, self.vz = vel
		self.dist = dist

	# ------------------------------- #

	def check_Vrel(self, minrange, maxrange):

		bool_list = []
		temp = np.array((1/self.dist) * ( (self.x * self.vx) + (self.y * self.vy) + (self.z * self.vz) ))

		for i in temp:
			if (minrange < i < maxrange):
				bool_list.append(True)
			else:
				bool_list.append(False)

		return bool_list

# ------------------------------- # ------------------------------- #

class Population:

	gayy = 1e-13 # GeV^-1
	r0 = 10 # km
	mass_NS = 1 # Msun
	rho_inf = 6.5 * 1e4 # 
	vDM = 200 # km/s (Dark Matter Velocity) 
	freq_factor = ( (1.6 * 1e-19) / (6.62607015 * 1e-34) ) * 1e-9
	c = 299792.458 # km/s

	beamFWHM = 1.24153/64 # axion_wavelength at 1 mueV / Parkes Survey Telescope dish aperture

	def __init__(self, B, P, alpha, pos, vel, dist, ma, time = 0):

		statement1 = "Order of parameters: B (in G), P (in seconds), alpha (in radiens),"
		statement2 = " co-ordinates(x, y, z) (in kpc), velocity(vx, vy, vz) (in km/s),"
		statement3 = " distance (in kpc), axion mass (in micro eV) and time (in Million years)."
		statement4 = " Permissible axion mass values - 0.1, 1, 10, 16.543 (4 GHz), 24.814 (6 GHz)"
		statement5 = ", 33.085 (8 GHz), 100."
		w.warn(statement1 + statement2 + statement3 + statement4 + statement5, SyntaxWarning, stacklevel = 2)

		self.B = np.array(B) # B0 in G
		self.P = np.array(P) # P in s
		self.alpha = np.array(alpha) 
		self.x, self.y, self.z = pos
		self.vx, self.vy, self.vz = vel
		self.dist = np.array(dist)
		self.ma = ma # in micro-eV
		self.time = time
		self.arrind = np.linspace(0, B.size - 1, B.size, dtype = int)

		if((self.B).size != (self.P).size or (self.P).size != (self.alpha).size):
			print((self.B).size, (self.P).size, (self.alpha).size)
			raise ValueError("Array sizes of B, P and alpha are not same.")

	# ------------------------------- #

	def check_Vrel(self, minrange, maxrange):

		bool_list = []
		temp = np.array((1/self.dist) * ( (self.x * self.vx) + (self.y * self.vy) + (self.z * self.vz) ))
		for i in temp:
			if (minrange < i < maxrange):
				bool_list.append(True)
			else:
				bool_list.append(False)

		return bool_list

	# ------------------------------- #

	def fobs(self):
		d = self.dist
		ma = self.ma * 1e-6 * self.freq_factor
		Vp = np.array([self.vx, self.vy, self.vz])
			# vx, vy, vz are from GC. As of now earth is stationary so Vp doesn't change.

		n_cap = np.array([self.x / d, (self.y - 8.5) / d, self.z / d])
			# n_cap is a unit vector from the sun (0, 8.5 ,0) to the pulsar

		newf = np.array([])

		Vrel = ( Vp[0] * n_cap[0] ) + ( Vp[1] * n_cap[1] ) + ( Vp[2] * n_cap[2] )
		newf = ma * (1 + ( Vrel/self.c ))

		return newf

	# ------------------------------- #

	def referenced_sorting(self, X, Y):
		Y_sort = np.array([x for _, x in sorted(zip(X, Y))]) # X dictates order
		return Y_sort

	# ------------------------------- #

	def allsky_intensity(self, doppf, L, bandwidth = 1e5):
		# L = self.power_hooke()
		d = self.dist
		delf = np.arange(np.min(doppf), np.max(doppf), bandwidth * 1e-9)
		if(delf[-1] != np.max(doppf)):
			delf = np.append(delf, np.max(doppf))

		if(delf.size < 1):
			raise ValueError("First find doppler shifted frequency, then use this function.")


		index = self.referenced_sorting(doppf, self.arrind)
		doppf_reord = np.sort(doppf)

		S = np.array([])
		num = np.array([], dtype = int)

		j, temp1, n  = 0, 0, 0
		for i in range(delf.size - 1):
			while(doppf_reord[j] <= delf[i+1]):
				temp1 = temp1 + (L[index[j]] / (4 * np.pi * d[index[j]] * d[index[j]] * bandwidth))
				j = j + 1
				n = n + 1

				if(j == index.size):
					break

			S = np.append(S, temp1)
			temp1 = 0
			num = np.append(num, n)
			n = 0

		S = (S / (3.086e19 * 3.086e19)) * 1e26 # convert into Jansky
		delf_center = delf[:-1] + np.diff(delf)/2

		intensity = S / (4 * np.pi) 

		return delf_center, intensity, num

	# ------------------------------- #

	def find_angle(self, target):
		vec1 = target
		vec2 = np.array([self.x, self.y - 8.5, self.z])
		num = (vec1[0] * vec2[0]) + (vec1[1] * vec2[1]) + (vec1[2] * vec2[2]) 
		cos_angle = num / ( np.linalg.norm(vec1) * np.linalg.norm(vec2, axis = 0) )
		angle = np.arccos(cos_angle)
		return angle

	# ------------------------------- #

	def targeted_intensity(self, doppf, angle, L, bandwidth = 1e5):
		# L = self.power_hooke()
		d = self.dist
		delf = np.arange(np.min(doppf), np.max(doppf), bandwidth * 1e9)
		if(delf[-1] != np.max(doppf)):
			delf = np.append(delf, np.max(doppf))

		if(delf.size < 1):
			raise ValueError("First find doppler shifted frequency, then use this function.")


		index = self.referenced_sorting(angle, self.arrind)
		doppf_refs = self.referenced_sorting(angle, doppf)
		angle_sort = np.sort(angle)

		S = np.zeros(delf.size - 1)
		num = np.zeros(delf.size - 1, dtype = int)

		j, temp1, n  = 0, 0, 0
		m = np.min(doppf)

		sigma = self.beamFWHM / (np.sqrt(8 * np.log(2)))

		while(angle_sort[j] < self.beamFWHM/2):
			iii = int( (doppf_refs[j] - m) / (bandwidth * 1e9) )
			weight =  np.exp(- angle_sort[j] * angle_sort[j] / (2 * sigma * sigma) )
			flux =  (L[index[j]] / (4 * np.pi * d[index[j]] * d[index[j]] * bandwidth)) * weight
			S[iii] = S[iii] + flux
			num[iii] = num[iii] + 1
			j = j+1


		S = (S / (3.086e19 * 3.086e19)) * 1e26 # convert into Jansky
		delf_center = delf[:-1] + np.diff(delf)/2

		intensity = S / (np.pi * self.beamFWHM * self.beamFWHM / 4)

		return delf_center, intensity, num

# ------------------------------- # ------------------------------- #

class Hooks_Analysis(Population):

	def power_hooke(self):

		# formula

		f_the_factor = ( (1.6 * 1e-19) / (6.62607015 * 1e-34) ) * 1e-9
		ma = self.ma * 1e-6 * self.freq_factor
		Omega = 2 * np.pi / self.P
		theta = np.random.random(size = (self.B).size) * (np.pi/2)

		m_dot_r = (cos(self.alpha) * cos(theta)) + (sin(self.alpha) * sin(theta) * cos(Omega * self.time))

				# Will remain constant
		part1 = 4.5 * 1e8 * ((self.gayy / 1e-12)**2) * ((self.r0/10)**2) * ((ma/1)**(5/3)) 
				# Will remain constant but can change later
		part2 = (self.mass_NS/1) * (200/self.vDM) * (self.rho_inf/0.3)
				# Is an array of values 
		part3 = ((self.B/1e14)**(2/3)) * ((self.P/1)**(4/3))
		part4 = ( (3 * m_dot_r * m_dot_r) + 1 ) / (np.abs( (3 * cos(theta) * m_dot_r) - cos(self.alpha) )**(4/3))

		return part1 * part2 * part3 * part4

# ------------------------------- # ------------------------------- #

class Surface_Element(Population):

	def interpolation(self):
		ma = self.ma
		Bval = self.B 
		Pval = self.P 
		alphaval = self.alpha
		if(ma == 0.1):
			with open("/Users/user/Desktop/Try/OOPS/interp_data_0.1.npy", "rb") as file:
				B, P, alpha, res = np.load(file), np.load(file), np.load(file), np.load(file)

		if(ma == 1):
			with open("/Users/user/Desktop/Try/OOPS/interp_data_1.npy", "rb") as file:
				B, P, alpha, res = np.load(file), np.load(file), np.load(file), np.load(file)

		if(ma == 10):
			with open("/Users/user/Desktop/Try/OOPS/interp_data_10.npy", "rb") as file:
				B, P, alpha, res = np.load(file), np.load(file), np.load(file), np.load(file)

		if(ma == 100):
			with open("/Users/user/Desktop/Try/OOPS/interp_data_100.npy", "rb") as file:
				B, P, alpha, res = np.load(file), np.load(file), np.load(file), np.load(file)

		if(ma == 16.543):
			with open("/Users/user/Desktop/Try/OOPS/interp_data_4GHz.npy", "rb") as file:
				B, P, alpha, res = np.load(file), np.load(file), np.load(file), np.load(file)

		if(ma == 24.814):
			with open("/Users/user/Desktop/Try/OOPS/interp_data_6GHz.npy", "rb") as file:
				B, P, alpha, res = np.load(file), np.load(file), np.load(file), np.load(file)

		if(ma == 33.085):
			with open("/Users/user/Desktop/Try/OOPS/interp_data_8GHz.npy", "rb") as file:
				B, P, alpha, res = np.load(file), np.load(file), np.load(file), np.load(file)

		res[res == 0] = 1
		interp = RegularGridInterpolator((np.log10(B), P, alpha), np.log10(res), method = "slinear")

		input_array = np.array([np.log10(Bval), Pval, alphaval]).T
		return 10**interp(input_array)




# ------------------------------- # ------------------------------- #



if __name__ == "__main__":
	getclass = Surface_Element()
	getclass.gen_data()
	B = 10 ** np.linspace(10, 14, 10)
	P = np.linspace(1, 5, 10)
	alpha = np.linspace(0, np.pi/2, 10)
	print(getclass.interpolation(B, P, alpha))