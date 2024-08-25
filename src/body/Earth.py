import math
import Body

class Earth(Body):
    """
    Earth-specific class that extends the generic Body class with Earth-specific properties and methods.
    """

    # Constants specific to Earth
    RADIUS_EQUATORIAL = 6378.137  # km
    RADIUS_POLAR = 6356.752  # km
    FLATTENING = 1 / 298.257223563  # Flattening factor
    GRAVITATIONAL_PARAMETER = 398600.4418  # km^3/s^2 (Earth's gravitational parameter, mu)
    ROTATION_RATE = 7.2921159e-5  # rad/s (Earth's rotational rate)
    MASS = 5.97219e24  # kg (Earth's mass)
    SURFACE_GRAVITY = 9.80665  # m/s^2 (Standard gravitational acceleration)

    def __init__(self):
        """
        Initialize the Earth object.
        """
        super().__init__(
            name="Earth",
            mass=Earth.MASS,
            radius=Earth.RADIUS_EQUATORIAL,  # Use the equatorial radius as a representative radius
            gravitational_parameter=Earth.GRAVITATIONAL_PARAMETER,
            rotation_rate=Earth.ROTATION_RATE
        )

    @staticmethod
    def mean_radius() -> float:
        """
        Get the mean radius of the Earth by averaging the equatorial and polar radii.
        
        :return: Mean radius of Earth in kilometers.
        """
        return (Earth.RADIUS_EQUATORIAL * 2 + Earth.RADIUS_POLAR) / 3

    @staticmethod
    def convert_to_geodetic(lat_geocentric: float) -> float:
        """
        Convert a geocentric latitude (measured from Earth's center) to geodetic latitude (measured from the surface).
        
        :param lat_geocentric: Geocentric latitude in degrees.
        :return: Geodetic latitude in degrees.
        """
        lat_geocentric_rad = math.radians(lat_geocentric)
        geodetic_latitude = math.atan2(math.tan(lat_geocentric_rad), (1 - Earth.FLATTENING) ** 2)
        return math.degrees(geodetic_latitude)

    @staticmethod
    def atmospheric_density(altitude: float) -> float:
        """
        Estimate the atmospheric density at a specific altitude above Earth's surface.
        This is a rough approximation using a simplified exponential atmosphere model.
        
        :param altitude: Altitude in kilometers above the Earth's surface.
        :return: Atmospheric density in kg/m^3.
        """
        if altitude < 0:
            raise ValueError("Altitude cannot be negative.")
        
        # Exponential model parameters (based on the US Standard Atmosphere)
        scale_height = 7.64  # km
        surface_density = 1.225  # kg/m^3 at sea level
        return surface_density * math.exp(-altitude / scale_height)


# Example Usage
if __name__ == "__main__":
    # Instantiate Earth
    earth = Earth()

    # Print Earth's information
    print(earth)

    # Calculate gravitational acceleration at 500 km altitude
    gravity_at_500km = earth.gravity_at_altitude(500)
    print(f"Gravitational acceleration at 500 km: {gravity_at_500km:.4f} m/s^2")

    # Calculate escape velocity at 500 km altitude
    escape_velocity_at_500km = earth.escape_velocity(500)
    print(f"Escape velocity at 500 km: {escape_velocity_at_500km:.4f} km/s")

    # Estimate atmospheric density at 100 km altitude
    atmospheric_density_100km = earth.atmospheric_density(100)
    print(f"Atmospheric density at 100 km: {atmospheric_density_100km:.6f} kg/m^3")
