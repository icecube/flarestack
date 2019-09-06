"""File containing links to data samples used (northern tracks).

Path to local copy of point source tracks, downloaded on 18/09/18 from
/data/ana .. /v002-p01, with the following readme:

    This folder containes datasets that are used in the point source analysis
    of the diffuse nu_mu-sample. The datasets correspond to 8 yr of data. These
    representcompressed versions of the files under
    /data/ana/analyses/northern_tracks/version-002-p00 with time, zen, azi
    fields added for time-dependent analysis.

    The sub-datasets are split due to different event selection. For each
    subsample you will find a file containing experimental events (exp) and
    one containig monte carlo events (MC).

    The files are labeled by "dataset" followed by version of diffuse fit
    followed by identifier for sub-dataset/event selection followed by
    identifier for experimental / monte carlo data and the date of creation.

    The all files have the following keys:

    * run				int - I3EventHeader Run ID
    * event				int - I3EventHeader Event ID
    * subevent			int - I3EventHeader SubEvent ID
    * azi				float  - zenith in radians
    * zen				float  - azimuth in radians
    * angErr			float  - pull corrected angular error in radians
    * time				double - MJD in days
    * ra				float  - R.A. in radians based on zen, azi and mjd
    * dec				float  - declination in radians based on zen, azi and mjd
    * logE				float  - log10 of energy_truncated / GeV

    In addition MC-files have the following keys:

    * trueRa	float  - true R.A. in radians of MCPrimary1
    * trueDec	float  - true declination in radians of MCPrimary1
    * trueE 	float  - true energy of MCPrimary1
    * conv		float  - event weight corresponding to
                            conventional component in diffuse fit
    * prompt	float  - event weight corresponding to prompt component
                            in diffuse fit
    * astro		float  - event weight correspinding to astrophysical component
                            in diffuse fit
    * ow    	double - one weight, calculated based on 'astro' of diffuse fit
                            but converted to OneWeight spectrum based on
                            diffuse best fit spectral index and
                            normalization, still includes effect of nuicience
                            parameters used in diffuse fit ow divided by
                            nfiles and nevents

    Livetimes in seconds:
        IC59        - 30079123.2
        IC79        - 26784000.0
        IC86-2011   - 29556432.0
        IC86-12-16  - 153228225.0
        (IC86-2012   - 28628486.0)
        (IC86-2013   - 31142758.0)
        (IC86-2014   - 31791941.0)
        (IC86-2015   - 31458917.0)
        (IC86-2016   - 30206123.0)

    Change logs:

    Version 2018/07/25
    * initial version



"""
from flarestack.data.icecube.ic_season import IceCubeDataset, \
    icecube_dataset_dir
from flarestack.data.icecube.northern_tracks import NTSeason, \
    get_diffuse_binning


nt_data_dir = icecube_dataset_dir + "northern_tracks/version-002-p05/"

nt_v002_p05 = IceCubeDataset()

sample_name = "northern_tracks_v002_p05"


def generate_diffuse_season(name):
    season = NTSeason(
        season_name=name,
        sample_name=sample_name,
        exp_path=nt_data_dir +
                 "dataset_8yr_fit_{0}_exp_compressed.npy".format(name),
        mc_path=nt_data_dir +
                "dataset_8yr_fit_{0}_MC_compressed.npy".format(name),
        grl_path=nt_data_dir +
                 "GRL/dataset_8yr_fit_{0}_exp_compressed.npy".format(name),
        sin_dec_bins=get_diffuse_binning(name)[0],
        log_e_bins=get_diffuse_binning(name)[1]
    )
    nt_v002_p05.add_season(season)


seasons = ["IC59", "IC79", "IC86_2011", "IC86_2012_16"]

for season in seasons:
    generate_diffuse_season(season)
