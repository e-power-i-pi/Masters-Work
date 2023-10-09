import numpy as np 
from scipy.interpolate import RegularGridInterpolator
import pandas as pd 
from multiprocess import Pool

# ------------------------------- # ------------------------------- #

class Surface_Element_Interp:

	def __init__(self):
		pass

# ------------------------------- #

	def gen_data(self):
		from Surface_Element import Surface_Element


		B = 10**np.linspace(9.8, 14.6, 80)
		P = np.linspace(10**-3, 10, 25)
		alpha = np.linspace(0, np.pi/2, 12)
		ma = np.array([0.1, 1, 10, 16.543, 24.814, 33.085, 100])
					# Corresponding 4GHz, 6GHz, 8GHz

		for iii in ma:
			def fnfn(bval, Pval, alphaval , maval = iii):
				x1 = Surface_Element(bval, Pval, maval, alphaval)
				return x1.calc_Power()

			with Pool() as p:
				result = np.array(p.starmap(fnfn,[(x,y,z) for x in B for y in P for z in alpha]))
				p.close() 
				p.join()

			res = np.reshape(result, (B.size, P.size, alpha.size))

			# Save it in the directory you want. These are names of the directory in my PC
			
			if(iii == 0.1):
				with open("/Users/user/Desktop/Try/OOPS/interp_data_0.1.npy", "wb") as file:
					np.save(file, B)
					np.save(file, P)
					np.save(file, alpha)
					np.save(file, res)

			if(iii == 1):
				with open("/Users/user/Desktop/Try/OOPS/interp_data_1.npy", "wb") as file:
					np.save(file, B)
					np.save(file, P)
					np.save(file, alpha)
					np.save(file, res)

			if(iii == 10):
				with open("/Users/user/Desktop/Try/OOPS/interp_data_10.npy", "wb") as file:
					np.save(file, B)
					np.save(file, P)
					np.save(file, alpha)
					np.save(file, res)

			if(iii == 100):
				with open("/Users/user/Desktop/Try/OOPS/interp_data_100.npy", "wb") as file:
					np.save(file, B)
					np.save(file, P)
					np.save(file, alpha)
					np.save(file, res)

			if(iii == 16.543):
				with open("/Users/user/Desktop/Try/OOPS/interp_data_4GHz.npy", "wb") as file:
					np.save(file, B)
					np.save(file, P)
					np.save(file, alpha)
					np.save(file, res)

			if(iii == 24.814):
				with open("/Users/user/Desktop/Try/OOPS/interp_data_6GHz.npy", "wb") as file:
					np.save(file, B)
					np.save(file, P)
					np.save(file, alpha)
					np.save(file, res)

			if(iii == 33.085):
				with open("/Users/user/Desktop/Try/OOPS/interp_data_8GHz.npy", "wb") as file:
					np.save(file, B)
					np.save(file, P)
					np.save(file, alpha)
					np.save(file, res)

		print("Data and result is ready to be interpolated")

# ------------------------------- #


	def interpolation(self, Bval, Pval, alphaval, ma):
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

# ------------------------------- #

if __name__ == "__main__":
	getclass = Surface_Element_Interp()
	getclass.gen_data()
	B = 10 ** np.linspace(10, 14, 10)
	P = np.linspace(1, 5, 10)
	alpha = np.linspace(0, np.pi/2, 10)
	ma = [0.1, 1, 10, 100, 16.543, 24.814, 33.085]

	for i in ma:
		print(getclass.interpolation(B, P, alpha, i))
