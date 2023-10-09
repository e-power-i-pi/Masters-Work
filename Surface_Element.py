import numpy as np
from numpy import sin, cos
from scipy.integrate import nquad
import matplotlib.pyplot as plt
import vegas
import psutil
from multiprocess import Pool 

class Surface_Element:

	def __init__(self, B, P, ma, mis_align):
		self.B = B
		self.P = P
		self.ma = ma
		self.mis_align = mis_align

	# ------------------------------- #

	def calc_Power(self):

		ma_mueV = self.ma # axion mass in micro eV

		''' 	Pulsar details '''
		Psec = self.P # period in seconds
		BGauss = self.B # magnetic field in Gauss
		theta_m = self.mis_align # mis-alignment in degrees


		''' 	ALL CONSTANTS '''
		''' 	Natural Units '''
		c = 299792458 # m/s
		hbar = 6.582119 * 1e-25 # GeV . s
		sec = (1/hbar) # GeV^-1
		m = (1/c) * sec # GeV^-1
		kg = (1/1.7827) * 1e27 # GeV

		''' 	Physical Constants '''
		Gauss = 1.95 * 1e-20 # GeV^2
		alphaEM = 1/137
		eEM = np.sqrt(4 * np.pi * alphaEM)
		Me = 0.5 * 1e-3 # GeV
		KM = 1e3 * m # GeV^-1
		micro_eV = 1e-6 * 1e-9 # GeV
		MNS = 1.9891 * 1e30 * kg # GeV
		G = 6.6743 * 1e-11 * (m / kg) * (m/sec) * (m/sec) # GeV^-2 
		gayy = 1e-10 # GeV^-1

		''' 	CONVERSION TO DIMENSIONLESS VARIABLES IN UNITS R=1 for 10 km NS '''
		me = Me * (10 * KM)
		omegaR = ( (2 * np.pi) / (Psec * sec) ) * (10 * KM)
		maR = (ma_mueV * micro_eV) * (10 * KM)
		beta = (10 * KM) * (10 * KM) * (BGauss * Gauss) # Magnetic Flux through a surface size RNS^2 
		RS = (2 * G * MNS) / (10 * KM)
		gayyR = gayy / (10 * KM)

		va = 220 * KM / sec # asymptotic axion velocity 220 km/sec
		# ma = ma_mueV * 1e-6 * 1e-9 # GeV
		GEV = 1 # eV
		cm = 0.01 * m # sec
		rho = 0.45 * GEV / (cm*cm*cm) # GeV^4

		rhoR = rho * ((10 * KM)**4)

		# -------------------------------------------------- #



		c1 = ( 2 * np.pi * alphaEM * beta * omegaR / (eEM * maR * maR * me) )**(1/3)

		rc = lambda t, theta, phi, alpha, Omega: c1 * (np.abs(  cos(alpha) + (3 * cos(alpha) * cos(2 * theta)) + (3 * cos(phi - (Omega * t)) * sin(alpha) * sin(2 * theta))  ) ** (1/3))



		nGJ = lambda t, theta, phi, r, alpha, Omega: beta * Omega * (cos(alpha) + 3*cos(alpha)*cos(2*theta) + 3*cos(phi - Omega*t)*sin(alpha)*sin(2*theta)) / (2 * eEM * r**3) 



		omegaP = lambda t, theta, phi, r, alpha, Omega: np.sqrt( 4 * np.pi * alphaEM * np.abs(nGJ(t, theta, phi, r, alpha, Omega)) / me )



		rho_inf = lambda r: rhoR / ( ( (1 - (RS/r))**(-1/2) ) * maR )  
		na = lambda r: rho_inf(r) * (2/np.sqrt(np.pi)) * (1/va) * np.sqrt( RS /r )



		c2 = maR * np.sqrt((1 + (va * va)))
		omegaC = lambda r: ((1 - (RS/r))**(-1/2)) * c2 



		vc = lambda r: np.sqrt(1 - ((maR*maR) / (omegaC(r) * omegaC(r))))



		modB = lambda t, theta, phi, r, alpha, Omega: (beta / ( 2*(r**3) )) * np.sqrt(1 + 3*(( cos(alpha)*cos(theta) + sin(alpha)*sin(theta)*cos(phi - t*Omega) )**2))


		hhh = 1e-7
		def grad_omegaP(t, theta, phi, r, alpha, Omega, key):
			hhh = 1e-7
			if(key == "r"):
				return (omegaP(t, theta, phi, r+hhh, alpha, Omega) - omegaP(t, theta, phi, r, alpha, Omega)) / hhh
			if(key == "t"):
				return (omegaP(t, theta+hhh, phi, r, alpha, Omega) - omegaP(t, theta, phi, r, alpha, Omega)) / hhh
			if(key == "p"):
				return (omegaP(t, theta, phi+hhh, r, alpha, Omega) - omegaP(t, theta, phi, r, alpha, Omega)) / hhh

		dot_grad_omegaP_r = lambda t, theta, phi, r, alpha, Omega: grad_omegaP(t, theta, phi, r, alpha, Omega, "r")**2
		dot_grad_omegaP_theta = lambda t, theta, phi, r, alpha, Omega: ((1/r) * grad_omegaP(t, theta, phi, r, alpha, Omega, "t"))**2
		dot_grad_omegaP_phi = lambda t, theta, phi, r, alpha, Omega: ((1/(r*sin(theta))) * grad_omegaP(t, theta, phi, r, alpha, Omega, "p"))**2
		dot_grad_omegaP = lambda t, theta, phi, r, alpha, Omega: np.sqrt(dot_grad_omegaP_r(t, theta, phi, r, alpha, Omega) + dot_grad_omegaP_theta(t, theta, phi, r, alpha, Omega) + dot_grad_omegaP_phi(t, theta, phi, r, alpha, Omega))



		Payy = lambda t, theta, phi, r, alpha, Omega, theta_k: (gayyR**2) * (sin(theta_k)**2) * (modB(t, theta, phi, r, alpha, Omega)**2) / (2 * dot_grad_omegaP(t, theta, phi, r, alpha, Omega))




		drc_dtheta = lambda t, theta, phi, alpha, Omega: (rc(t, theta + hhh, phi, alpha, Omega) - rc(t, theta, phi, alpha, Omega)) / hhh 

		drc_dphi = lambda t, theta, phi, alpha, Omega: (rc(t, theta, phi + hhh, alpha, Omega) - rc(t, theta, phi, alpha, Omega)) / hhh



		A_const = lambda r: 1 - (RS/r)

		c3 = 2 * np.pi 
		fa = lambda r: vc(r) * maR * na(r) / (4 * np.pi)
		d3k = lambda t, theta, phi, r, alpha, Omega: r * np.sqrt( (A_const(r)*((r*sin(theta))**2)) + ((drc_dtheta(t, theta, phi, alpha, Omega) * sin(theta))**2) + ((drc_dphi(t, theta, phi, alpha, Omega))**2) )


		Power = lambda t, theta, phi, r, alpha, Omega, theta_k: (c3 * sin(theta_k) * fa(r) * Payy(t, theta, phi, r, alpha, Omega, theta_k) * d3k(t, theta, phi, r, alpha, Omega) if r >= 1 else 0)


		# -------------------------------------------------- #

		THETA = np.linspace(0, np.pi, 1000)

		def fn(x):
			theta, phi, theta_k = x
			return Power(0, theta, phi, rc(0, theta, phi, theta_m, omegaR), theta_m, omegaR, theta_k)

		integ = vegas.Integrator([[0, np.pi], [0, 2*np.pi], [0, np.pi]])

		integ(fn, nitn = 5, neval = 1000)

		result = integ(fn, nitn = 10, neval = 5000)

		return (result * 1e9 / (hbar * (10 * KM) * (10 * KM))).mean
