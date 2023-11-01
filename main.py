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
sim_duration = 60.0 * 60 * 24 * 30 # seconds
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
# coesSun = [149.598e6, 0.0167, 23.4406, 0, 282.7685, 0]
# orbitSun = Orbit(coesSun)
# sun = Satellite(1.989e30, orbitSun)
# propagateSun = Propagate(sun, sim_duration, dt)
# statesSun, statesGeocSun = propagateSun.propagate(t0, options={'j2': False})
# print(statesSun)
# print(statesSat)

# Check for eclipses
# stateEclipse, sunDot, perpNorm = in_eclipse(statesSat, statesSun)
# file_name = 'eclipse.txt'
# file = open(file_name, 'w')

# for (i, state) in enumerate(stateEclipse):
#     if (state):
#         print(f'In eclipse at {datetime.fromtimestamp(t0 + i * dt)}')
#         file.write(f'In eclipse at {datetime.fromtimestamp(t0 + i * dt)}; {sunDot[i]}; {perpNorm[i]}\n')
# file.close()

# Get data from STK
stk_file_name = 'Satellite1_Classical_Orbit_Elements.csv'
stk_data = []
stk_data_file = open(stk_file_name, newline='')
stk_file_reader = csv.reader(stk_data_file)
stkStates = np.zeros((len(stk_data), 6)) # Actual state vector of STK satellite

# Skip the header row if it exists
next(stk_file_reader, None)
for row in stk_file_reader:
    stk_data.append(row)

# Read each row in the CSV file
for i in range(len(stk_data)):
    row = stk_data[i]
    stateStk = Orbit.Coes2State([float(row[1]), float(row[2]), float(row[3]), float(row[6]), float(row[5]), float(row[4])])
    stkStates[i] = stateStk


plot_groundtracks([stkStatesGeoc, statesSat])
plot_eci([stkStates, statesSat], {'show': True, 'cb_axes_color': 'k'})
# plt.show()
