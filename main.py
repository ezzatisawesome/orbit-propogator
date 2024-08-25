from datetime import datetime 
import matplotlib.pyplot as plt

from src.orbit import Orbit
from src.bodies import Earth
from src.dataclasses import ClassicalOrbitalElements
from src.plot import plot_eci


# Epoch (Vernal Equinox 2024)
year = 2024
month = 3
day = 20
hour = 6 
minute = 0
second = 0
t0 = datetime(year, month, day, hour, minute, second)


# Satellite orbit
coesSat = ClassicalOrbitalElements(7641.80, 0.00000001, 100.73, 0, 0, 90)
orbit = Orbit.from_coes(coesSat, Earth, t0)

# Output:
statesSat = []
statesGeocSat = []

# Propagate 1 minute
for i in range(3600):
    states, statesGeoc = orbit.propagate(1, 1)
    statesSat.append(states[0])
    statesGeocSat.append(statesGeoc[0])
    print(type(states))


# Plot the satellite trajectory
print(statesSat)
plot_eci(
    [statesSat],
    {
        "cb_axes_color": "k",
        "opacity": 0.5,
        "figsize": (20, 10),
        "title": "Satellite Orbit"
    },
)
plt.show()