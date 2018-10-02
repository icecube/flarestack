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

from flarestack.shared import dataset_dir
import numpy as np
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_data_dir

nt_data_dir = dataset_dir + "northern_tracks/version-002-p01/"

diffuse_dict = {
    "Data Sample": "diffuse_8_year",
    "sinDec bins": np.unique(np.concatenate([
        np.linspace(-0.05, 0.2, 8 + 1),
        np.linspace(0.2, 0.9, 12 + 1),
        np.linspace(0.9, 1., 2 + 1),
    ])),
    "MJD Time Key": "time"
}

diffuse_IC59 = {
    "Name": "IC59",
    "exp_path": nt_data_dir + "dataset_8yr_fit_IC59_exp_compressed.npy",
    "mc_path": nt_data_dir + "dataset_8yr_fit_IC59_MC_compressed.npy",
    "grl_path": ps_data_dir + "IC59_GRL.npy"
}
diffuse_IC59.update(diffuse_dict)

diffuse_IC79 = {
    "Name": "IC79",
    "exp_path": nt_data_dir + "dataset_8yr_fit_IC79_exp_compressed.npy",
    "mc_path": nt_data_dir + "dataset_8yr_fit_IC79_MC_compressed.npy",
    "grl_path": nt_data_dir + "GRL/GRL_IC79.npy"
}
diffuse_IC79.update(diffuse_dict)

diffuse_IC86_1 = {
    "Name": "IC86_1",
    "exp_path": nt_data_dir + "dataset_8yr_fit_IC86_2011_exp_compressed.npy",
    "mc_path": nt_data_dir + "dataset_8yr_fit_IC86_2011_MC_compressed.npy",
    "grl_path": nt_data_dir + "GRL/GRL_IC86_2011.npy"
}
diffuse_IC86_1.update(diffuse_dict)

diffuse_IC86_23456 = {
    "Name": "IC86_23456",
    "exp_path": nt_data_dir + "dataset_8yr_fit_IC86_2012_16_exp_compressed.npy",
    "mc_path": nt_data_dir + "dataset_8yr_fit_IC86_2012_16_MC_compressed.npy",
    "grl_path": nt_data_dir + "GRL/GRL_IC86_2012_16.npy"
}

diffuse_IC86_23456.update(diffuse_dict)

diffuse_8year = [diffuse_IC59, diffuse_IC79, diffuse_IC86_1, diffuse_IC86_23456]
