import matplotlib.pyplot as plt
from datetime import datetime

from src.orbit import Orbit
from src.satellite import Satellite
from src.propagate import Propagate
from src.plot import plot_groundtracks, plot_eci

# Simulation parameters
sim_duration = 60.0 * 60 * 24  # seconds
dt = 100

# Epoch (Vernal Equinox 2024)
# year = 2024
# month = 3
# day = 20
# hour = 6 #! Check w/ Troy for times of a dusk/dawn orbit
# minute = 0
# second = 0
# t0 = datetime(year, month, day, hour, minute, second).timestamp()

# Satellite orbit
coesSat = [7641.80, 0.0, 98.94, 0, 0, 0]  # sma, ecc, inc, ta, aop, raan, t
orbit = Orbit(coesSat)
satellite = Satellite(1400, orbit)  # ! Need to fix satellite mass
propagateSat = Propagate(satellite, sim_duration, dt)
statesSat, statesGeocSat = propagateSat.propagate(0, options={'j2': True})

# Sun orbit
# coesSun = [149.60e6, 0.0167, 23.44, 0, 288.1, 0]
# orbitSun = Orbit(coesSun)
# sun = Satellite(1.989e30, orbitSun)
# propagateSun = Propagate(sun, sim_duration, dt)
# statesSun, statesGeocSun = propagateSun.propagate(t0, options={'j2': False})

# Check for eclipses
# stateEclipse = in_eclipse(statesSat, statesSun)
# print(stateEclipse)

plot_groundtracks(statesSat)
plot_eci([statesGeocSat], {'show': True, 'cb_axes_color': 'k'})
plt.show()
