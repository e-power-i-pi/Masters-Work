import numpy as np
import matplotlib.pyplot as plt

def theta(r, k, r0, theta0):
	return k * np.log(r/r0) + theta0


r = np.linspace(2, 30, 1000)

k = [4.25, 4.25, 4.89, 4.89]
r0 = [3.48, 3.48, 4.90, 4.90]
theta0 = [1.57, 4.71, 4.09, 0.95]

fig = plt.figure()
ax = fig.add_subplot(facecolor = "black")
axp = fig.add_axes(ax.get_position().bounds, polar = True, frameon = False)

s = np.linspace(40, 0.1, 1000)

axp.scatter(theta(r, k[0], r0[0], theta0[0]), r, color = "white", s = s, alpha = 0.6)
ax.text(-10, -27, "Norma", color = "#fdffd6")

axp.scatter(theta(r, k[1], r0[1], theta0[1]), r, color = "white", s = s, alpha = 0.6)
ax.text(-12, 27, "Carina-Sagittarius", color = "#bfffc1")

axp.scatter(theta(r, k[2], r0[2], theta0[2]), r, color = "white", s = s, alpha = 0.6)
ax.text(20, -17, "Perseus", color = "#ffe1ea")

axp.scatter(theta(r, k[3], r0[3], theta0[3]), r, color = "white", s = s, alpha = 0.6)
ax.text(-28, 20, "Crux-Scutum", color = "white")


axp.scatter(0, 0, color = "white", s = 2800)
axp.scatter(0, 0, color = "black", s = 20, label = "Sgr A*")
axp.scatter(0, 8.5, color = "orange", s = 20, label = "Sun", marker = r"$\odot$")

ax.set_aspect('equal')
ax.set_xlim(-30,30)
ax.set_ylim(-30,30)
axp.set_rlim(0,30)
axp.grid()
axp.spines['polar'].set_visible(False)
axp.set_yticklabels([])
axp.set_xticklabels([])
ax.grid(alpha = 0.2)
plt.legend(loc = 4, fontsize = 9)
plt.rcParams["font.family"] = "serif"
ax.set_xlabel("X (kpc)")
ax.set_ylabel("Y (kpc)")
plt.savefig("/Users/user/Desktop/Results/Thesis/Chapter 3/galaxy_spirals.png", 
	bbox_inches = "tight", dpi = 300)

plt.show()