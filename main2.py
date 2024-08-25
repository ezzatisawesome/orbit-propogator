from datetime import datetime 

from src.orbit import Orbit
from src.bodies import Earth
from src.dataclasses import ClassicalOrbitalElements


# Epoch (Vernal Equinox 2024)
year = 2024
month = 3
day = 20
hour = 6 
minute = 0
second = 0
t0 = datetime(year, month, day, hour, minute, second).timestamp()


# Satellite orbit
coesSat = ClassicalOrbitalElements(7641.80, 0.00000001, 100.73, 0, 0, 90)
orbit = Orbit.from_coes(coesSat, Earth, t0)
print(orbit.propagate(1, 1))  # Propagate 1 hour