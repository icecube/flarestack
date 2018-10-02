"""File containing links to data samples used (pointsource tracks).

Path to local copy of point source tracks, downloaded on 24/04/18 from
/data/ana .. /current, with the following readme:

    This directory contains a patched version of Stefan Coenders' original npy
    files prepared for the 7yr time integrated paper (version-002p00).
    IC40 through IC86 2011 are exactly the same as used in the paper.

    The files contain some track events which overlap with the MESE sample,
    but not all MESE events are present. Do not use these file if you want
    to perform MESE + PS style analysis.

    IC86 2012-2014 has one small patch to fix a known bug in the original files:

    FIX - updated per event angular uncertainties for IC86 2012-2014 files

    This fix is done by

    (1) applying median angular resolution from bootstrap method in cases
    where the sigma paraboloid fit fails
    (2) apply the pull correction splines from Asen's time dependent
    unblinding as they are currently our best pull correction of
    the 7yr sample. See: https://docushare.icecube.wisc.edu/dsweb/Get/Document-
    77805/christov_PSCall_27.6.2016.pdf

    Below is a description of all the years and their corresponding files.

    Josh - Nov 6, 2017

    IC40:
      Data File  IC40_exp.npy
      MC File    IC40_corrected_MC.npy

    IC59:
      Data File  IC59_exp.npy
      MC File    IC59_corrected_MC.npy

    IC79:
      Data File  IC79b_exp.npy
      MC File    IC79b_corrected_MC.npy

    IC86, 2011:
      Data File  IC86_exp.npy
      MC File    IC86_corrected_MC.npy

    IC86, 2012:
      Data File  IC86-2012_exp_v2.npy
      MC File    IC86-2012_corrected_MC_v2.npy

    IC86, 2013:
      Data File  IC86-2013_exp_v2.npy
      MC File    IC86-2013_corrected_MC_v2.npy

    IC86, 2014:
      Data File  IC86-2014_exp_v2.npy
      MC File    IC86-2014_corrected_MC_v2.npy

"""
from flarestack.shared import dataset_dir
import numpy as np

ps_data_dir = dataset_dir + "ps_tracks/version-002-p01/"

ps_dict = {
    "Data Sample": "ps_tracks_v002_p01",
    "sinDec bins": np.unique(np.concatenate([
            np.linspace(-1., -0.9, 2 + 1),
            np.linspace(-0.9, -0.2, 8 + 1),
            np.linspace(-0.2, 0.2, 15 + 1),
            np.linspace(0.2, 0.9, 12 + 1),
            np.linspace(0.9, 1., 2 + 1),
        ])),
    "MJD Time Key": "time"
}

IC40_dict = {
    "Name": "IC40",
    "exp_path": ps_data_dir + "IC40_exp.npy",
    "mc_path": ps_data_dir + "IC40_corrected_MC.npy",
    "grl_path": ps_data_dir + "IC40_GRL.npy"
}
IC40_dict.update(ps_dict)

IC59_dict = {
    "Name": "IC59",
    "exp_path": ps_data_dir + "IC59_exp.npy",
    "mc_path": ps_data_dir + "IC59_corrected_MC.npy",
    "grl_path": ps_data_dir + "IC59_GRL.npy"
}
IC59_dict.update(ps_dict)


IC79_dict = {
    "Name": "IC79",
    "exp_path": ps_data_dir + "IC79b_exp.npy",
    "mc_path": ps_data_dir + "IC79b_corrected_MC.npy",
    "grl_path": ps_data_dir + "IC79b_GRL.npy"
}
IC79_dict.update(ps_dict)


IC86_1_dict = {
    "Name": "IC86_1",
    "exp_path": ps_data_dir + "IC86_exp.npy",
    "mc_path": ps_data_dir + "IC86_corrected_MC.npy",
    "grl_path": ps_data_dir + "IC86_GRL.npy"
}
IC86_1_dict.update(ps_dict)


IC86_234_dict = {
    "Name": "IC86_234",
    "exp_path": [
        ps_data_dir + "IC86-2012_exp_v2.npy",
        ps_data_dir + "IC86-2013_exp_v2.npy",
        ps_data_dir + "IC86-2014_exp_v2.npy"
        ],
    "mc_path": ps_data_dir + "IC86-2012_corrected_MC_v2.npy",
    "grl_path": [
        ps_data_dir + "IC86-2012_GRL.npy",
        ps_data_dir + "IC86-2013_GRL.npy",
        ps_data_dir + "IC86-2014_GRL.npy"
    ]
}

IC86_234_dict.update(ps_dict)

ps_7year = [
    IC40_dict, IC59_dict, IC79_dict, IC86_1_dict, IC86_234_dict,
]