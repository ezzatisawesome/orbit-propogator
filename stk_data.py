import numpy as np
import csv

from src.orbit import Orbit
from src.constants import earth_mu

def getStkData():
    # Setup CSV reader
    stk_file_name = 'Satellite1_Classical_Orbit_Elements.csv'
    stk_data = []
    stk_data_file = open(stk_file_name, newline='')
    stk_file_reader = csv.reader(stk_data_file)

    # Get CSV data
    next(stk_file_reader, None)
    for row in stk_file_reader:
        stk_data.append(row)
    stkStates = np.zeros((len(stk_data), 6)) # Actual state vector of STK satellite

    # Read each row in the CSV file
    for i in range(len(stk_data)):
        row = stk_data[i]
        stateStk = Orbit.Coes2State(
            [float(row[1]), float(row[2]), float(row[3]), float(row[6]), float(row[5]), float(row[4])],
            earth_mu
        )
        stkStates[i] = stateStk

    return stkStates

if __name__ == '__main__':
    getStkData()