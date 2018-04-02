import numpy as np
from signal_over_background import SoB
from time_PDFs import TimePDF


class LLH(SoB):
    """General  LLH class.
    """

    def __init__(self, season, sources, **kwargs):
        print "Initialising LLH for", season["Name"]
        SoB.__init__(self, season, **kwargs)
        self.sources = sources
        self.time_PDF = TimePDF.create(kwargs["LLH Time PDF"],
                                       season)

    def select_coincident_data(self, data, sources):
        """Checks each source, and only identifies events in data which are
        both spatially and time-coincident with the source. Spatial
        coincidence is defined as a +/- 5 degree box centered on the  given
        source. Time coincidence is determined by the parameters of the LLH
        Time PDF. Produces a mask for the dataset, which removes all events
        which are not coincident with at least one source.

        :param data: Dataset to be tested
        :param sources: Sources to be tested
        :return: Mask to remove
        """
        veto = np.ones_like(data["timeMJD"], dtype=np.bool)

        for source in sources:
            # Sets time mask, based on parameters for LLH Time PDF
            start_time, end_time = self.time_PDF.flare_time_mask(source)
            time_mask = np.logical_and(np.greater(data["timeMJD"], start_time),
                                       np.less(data["timeMJD"], end_time))

            # Sets half width of spatial box
            width = np.deg2rad(5.)

            # Sets a declination band 5 degrees above and below the source
            min_dec = max(-np.pi / 2., source['dec'] - width)
            max_dec = min(np.pi / 2., source['dec'] + width)

            # Accepts events lying within a 5 degree band of the source
            dec_mask = np.logical_and(np.greater(data["dec"], min_dec),
                                      np.less(data["dec"], max_dec))

            # Sets the minimum value of cos(dec)
            cos_factor = np.amin(np.cos([min_dec, max_dec]))

            # Scales the width of the box in ra, to give a roughly constant
            # area. However, if the width would have to be greater that +/- pi,
            # then sets the area to be exactly 2 pi.

            dPhi = np.amin([2. * np.pi, 2. * width / cos_factor])

            # Accounts for wrapping effects at ra=0, calculates the distance
            # of each event to the source.

            ra_dist = np.fabs(
                (data["ra"] - source['ra'] + np.pi) % (2. * np.pi) - np.pi)
            ra_mask = ra_dist < dPhi / 2.

            spatial_mask = dec_mask & ra_mask

            coincident_mask = spatial_mask & time_mask

            veto = veto & ~coincident_mask

        print "Of", len(data), "events, we consider", np.sum(~veto), "events."
        return ~veto

    def find_flare(self, data):
        pass
