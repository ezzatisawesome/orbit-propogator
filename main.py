import matplotlib.pyplot as plt
from datetime import datetime
import csv
import numpy as np

from src.orbit import Orbit
from src.satellite import Satellite
from src.propagate import Propagate
from src.plot import plot_groundtracks, plot_eci
from src.utils import in_eclipse
from src.constants import earth_radius

# Simulation parameters
sim_duration = 60.0 * 60 * 24 * 365# seconds
dt = 10

# Epoch (Vernal Equinox 2024)
year = 2024
month = 3
day = 20
hour = 6 #! Check w/ Troy for times of a dusk/dawn orbit
minute = 0
second = 0
t0 = datetime(year, month, day, hour, minute, second).timestamp()

# Satellite orbit
coesSat = [7641.80, 0.00000001, 100.73, 0, 0, 90]  # sma, ecc, inc, ta, aop, raan
orbit = Orbit(coesSat)
satellite = Satellite(1400, orbit)  # ! Need to fix satellite mass
propagateSat = Propagate(satellite, sim_duration, dt)
statesSat, statesGeocSat = propagateSat.propagate(t0, options={'j2': True})

# Sun orbit
coesSun = [149.598e6, 0.0000001, 23.4406, 0, 0, 0]
orbitSun = Orbit(coesSun)
sun = Satellite(1.989e30, orbitSun, options={'sun': True})
propagateSun = Propagate(sun, sim_duration, dt)
statesSun, statesGeocSun = propagateSun.propagate(t0, options={'j2': False})

# Check for eclipses
stateEclipse, sunDot, perpNorm = in_eclipse(statesSat, statesSun)
file_name = 'eclipse.txt'
file = open(file_name, 'w')
for (i, state) in enumerate(stateEclipse):
    if (state):
        # print(f'In eclipse at {datetime.fromtimestamp(t0 + i * dt)}')
        file.write(f'In eclipse at {datetime.fromtimestamp(t0 + i * dt)}; {sunDot[i]}; {perpNorm[i]}\n')
file.close()

# Plot
plot_groundtracks([statesGeocSat, statesGeocSun])
plot_eci([statesSat, statesSun], {'show': True, 'cb_axes_color': 'k'})
plt.show()
