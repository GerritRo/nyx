import astropy.units as u
from astropy.coordinates import FunctionTransform, BaseCoordinateFrame, RepresentationMapping, frame_transform_graph
from astropy.coordinates import SphericalRepresentation, GeocentricTrueEcliptic, TimeAttribute, get_body

class SunRelativeEclipticFrame(BaseCoordinateFrame):
    default_representation = SphericalRepresentation

    # Declare required frame attributes
    obstime = TimeAttribute(default=None)

    frame_specific_representation_info = {
        SphericalRepresentation: [
            RepresentationMapping('lon', 'alpha'),
            RepresentationMapping('lat', 'beta'),
            RepresentationMapping('distance', 'distance')
        ]
    }

@frame_transform_graph.transform(FunctionTransform,
                                 GeocentricTrueEcliptic, SunRelativeEclipticFrame)
def gte_to_sunrel(gte_coords, sunrel_frame):
    obstime = gte_coords.obstime
    if obstime is None:
        raise ValueError("GeocentricTrueEcliptic coords must have obstime")

    # Get Sun in the same frame
    sun = get_body("sun", obstime)
    sun_ecl = sun.transform_to(GeocentricTrueEcliptic(obstime=obstime))

    alpha = (gte_coords.lon - sun_ecl.lon).wrap_at(180 * u.deg)
    beta = gte_coords.lat
    distance = gte_coords.distance if gte_coords.distance.unit != u.one else None

    return SunRelativeEclipticFrame(alpha=alpha, beta=beta, distance=distance, obstime=obstime)

@frame_transform_graph.transform(FunctionTransform,
                                 SunRelativeEclipticFrame, GeocentricTrueEcliptic)
def sunrel_to_gte(sunrel_coords, gte_frame):
    obstime = sunrel_coords.obstime
    if obstime is None:
        raise ValueError("SunRelativeEclipticFrame must have obstime")

    # Get Sun's ecliptic coords
    sun_ecl = get_body("sun", obstime).transform_to(GeocentricTrueEcliptic(obstime=obstime))

    lon = (sun_ecl.lon + sunrel_coords.alpha).wrap_at(360 * u.deg)
    lat = sunrel_coords.beta
    distance = sunrel_coords.distance if sunrel_coords.distance.unit != u.one else None

    from astropy.coordinates import SkyCoord
    return SkyCoord(lon=lon, lat=lat, distance=distance,
                    frame=GeocentricTrueEcliptic(obstime=obstime))