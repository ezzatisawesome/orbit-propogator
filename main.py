from src.orbit import Orbit
from src.satellite import Satellite
from src.propagate import Propagate


coes = [7641.80, 0.0, 98.94, 0, 0, 0] # sma, ecc, inc, ta, aop, raan
sim_duration = 60.0 * 250  # seconds

orbit = Orbit(coes)
satellite = Satellite(1400, orbit) #! Need to fix satellite mass
propagate = Propagate(satellite, sim_duration, 50)

propagate.propagate()