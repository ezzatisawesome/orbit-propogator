import os
import matplotlib.pyplot as plt
import numpy as np

COLORS = [
    'm', 'deeppink', 'chartreuse', 'w', 'springgreen', 'peachpuff',
    'white', 'lightpink', 'royalblue', 'lime', 'aqua'] * 100

COASTLINES_COORDINATES_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.path.join('data', 'coastlines.csv')
)

EARTH_SURFACE_IMAGE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.path.join('data', 'earth_surface.png')
)

SURFACE_BODY_MAP = {
    'earth': EARTH_SURFACE_IMAGE,
}


def plot_groundtracks(geocentric_states: np.ndarray[float]):
    plt.figure(figsize=(16, 8))
    coast_cords = np.genfromtxt(COASTLINES_COORDINATES_FILE, delimiter=',')
    plt.plot(coast_cords[:, 0], coast_cords[:, 1], 'mo', markersize=0.3)

    plt.plot(
        geocentric_states[:, 0],
        geocentric_states[:, 1],
        color='red',
        marker='o',
        linestyle='none',
    )

    plt.imshow(
        plt.imread(EARTH_SURFACE_IMAGE),
        extent=[-180, 180, -90, 90])


if __name__ == "__main__":
    plot_groundtracks()
    plt.show()
