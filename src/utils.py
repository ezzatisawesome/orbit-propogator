import math

def Cal2Gmst(Y1, M1, D1, D):
    # Compute modified month and year
    if M1 <= 2:
        Y2 = Y1 - 1
        M2 = M1 + 12
    else:
        Y2 = Y1
        M2 = M1

    B = Y1 / 400 - Y1 / 100 + Y1 / 4

    # Decimal days
    D2 = D1 + D

    # Modified Julian Date
    MJD = 365 * Y2 - 679004 + int(B) + int(30.6001 * (M2 + 1)) + D2
    d = MJD - 51544.5

    # GMST in degrees
    GMST = 280.4606 + 360.9856473 * d
    GMST = math.radians(GMST) # Convert to radians
    GMST = GMST % (2 * math.pi) # Ensure GMST is in the range [0, 2 * pi)

    return GMST