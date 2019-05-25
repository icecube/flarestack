import numpy as np
import healpy as hp
from scipy.stats import norm
from numpy.lib.recfunctions import append_fields
from flarestack.core.astro import angular_distance


class SpatialPDF:

    subclasses = {}

    def __init__(self, spatial_pdf_dict):
        pass

    @classmethod
    def register_subclass(cls, spatial_pdf_name):
        """Adds a new subclass of SpatialPDF, with class name equal to
        "spatial_pdf_name".
        """
        def decorator(subclass):
            cls.subclasses[spatial_pdf_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, s_pdf_dict):

        try:
            s_pdf_name = s_pdf_dict["spatial_pdf_name"]
        except KeyError:
            s_pdf_name = "circular_gaussian"

        if s_pdf_name not in cls.subclasses:
            raise ValueError('Bad spatial PDF name {}'.format(s_pdf_name))

        return cls.subclasses[s_pdf_name](s_pdf_dict)

    @staticmethod
    def simulate_distribution(source, data):
        return data

    @staticmethod
    def signal_spatial(source, events):
        return

    @staticmethod
    def rotate(ra1, dec1, ra2, dec2, ra3, dec3):
        """Rotate ra1 and dec1 in a way that ra2 and dec2 will exactly map
        onto ra3 and dec3, respectively. All angles are treated as radians.
        Essentially rotates the events, so that they behave as if they were
        originally incident on the source.

        :param ra1: Event Right Ascension
        :param dec1: Event Declination
        :param ra2: True Event Right Ascension
        :param dec2: True Event Declination
        :param ra3: Source Right Ascension
        :param dec3: Source Declination
        :return: Returns new Right Ascensions and Declinations
        """
        # Turns Right Ascension/Declination into Azimuth/Zenith for healpy
        phi1 = ra1 - np.pi
        zen1 = np.pi/2. - dec1
        phi2 = ra2 - np.pi
        zen2 = np.pi/2. - dec2
        phi3 = ra3 - np.pi
        zen3 = np.pi/2. - dec3

        # Rotate each ra1 and dec1 towards the pole?
        x = np.array([hp.rotator.rotateDirection(
            hp.rotator.get_rotation_matrix((dp, -dz, 0.))[0], z, p)
            for z, p, dz, dp in zip(zen1, phi1, zen2, phi2)])

        # Rotate **all** these vectors towards ra3, dec3 (source_path)
        zen, phi = hp.rotator.rotateDirection(np.dot(
            hp.rotator.get_rotation_matrix((-phi3, 0, 0))[0],
            hp.rotator.get_rotation_matrix((0, zen3, 0.))[0]), x[:, 0], x[:, 1])

        dec = np.pi/2. - zen

        ra = phi + np.pi
        return np.atleast_1d(ra), np.atleast_1d(dec)

    def rotate_to_position(self, ev, ra, dec):
        """Modifies the events by reassigning the Right Ascension and
        Declination of the events. Rotates the events, so that they are
        distributed as if they originated from the source. Removes the
        additional Monte Carlo information from sampled events, so that they
        appear like regular data.

        The fields removed are:
            True Right Ascension,
            True Declination,
            True Energy,
            OneWeight

        :param ev: Events
        :param ra: Source Right Ascension (radians)
        :param dec: Source Declination (radians)
        :return: Events (modified)
        """
        names = ev.dtype.names

        # Rotates the events to lie on the source
        ev["ra"], rot_dec = self.rotate(ev["ra"], np.arcsin(ev["sinDec"]),
                                        ev["trueRa"], ev["trueDec"],
                                        ra, dec)

        if "dec" in names:
            ev["dec"] = rot_dec
        ev["sinDec"] = np.sin(rot_dec)

        # Deletes the Monte Carlo information from sampled events
        non_mc = [name for name in names
                  if name not in ["trueRa", "trueDec", "trueE", "ow"]]
        ev = ev[non_mc].copy()

        return ev


@SpatialPDF.register_subclass("circular_gaussian")
class CircularGaussian(SpatialPDF):

    def simulate_distribution(self, source, data):
        data["ra"] = np.pi + norm.rvs(size=len(data)) * data["sigma"]
        data["dec"] = norm.rvs(size=len(data)) * data["sigma"]
        data["sinDec"] = np.sin(data["dec"])
        data = append_fields(
            data, ["trueRa", "trueDec"],
            [np.ones_like(data["dec"]) * np.pi, np.zeros_like(data["dec"])]
        ).copy()
        
        data = self.rotate_to_position(
            data, source["ra_rad"], source["dec_rad"]).copy()

        return data.copy()

    @staticmethod
    def signal_spatial(source, cut_data):
        """Calculates the angular distance between the source and the
        coincident dataset. Uses a Gaussian PDF function, centered on the
        source. Returns the value of the Gaussian at the given distances.

        :param source: Single Source
        :param cut_data: Subset of Dataset with coincident events
        :return: Array of Spatial PDF values
        """
        distance = angular_distance(
            cut_data['ra'], cut_data['dec'],
            source['ra_rad'], source['dec_rad']
        )
        space_term = (1. / (2. * np.pi * cut_data['sigma'] ** 2.) *
                      np.exp(-0.5 * (distance / cut_data['sigma']) ** 2.))

        return space_term

