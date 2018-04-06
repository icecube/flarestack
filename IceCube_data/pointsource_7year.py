data_dir = "/afs/ifh.de/user/a/astasik/scratch/PS_Data"

IC79_dict = {
    "Name": "IC79",
    "exp_path": data_dir + "/FinalSample/IC79/exp/IC79_exp_corrected.npy",
    "mc_path": data_dir + "/FinalSample/IC79/mc/IC79_nugen_corrected.npy",
    "aw_path": data_dir + "/DeclinationAcceptance/IC79",
    "Livetime": 315.506,
    "Start (MJD)": 55347.2862153,
    "End (MJD)": 55694.4164699
}

IC86_1_dict = {
    "Name": "IC86_1",
    "exp_path": data_dir + "/FinalSample/IC86_1/exp/IC86_1_exp_corrected.npy",
    "mc_path": data_dir + "/FinalSample/IC86_1/mc/IC86_1_nugen_corrected.npy",
    "aw_path": data_dir + "/DeclinationAcceptance/IC86_1",
    "Livetime": 332.61,
    "Start (MJD)": 55694.4164699,
    "End (MJD)": 56062.420706
}

IC86_234_dict = {
    "Name": "IC86_234",
    "exp_path": data_dir + "/FinalSample/IC86_2AndFollowing/exp"
                 "/IC86_2_3_4_exp_corrected.npy",
    "mc_path": data_dir + "/FinalSample/IC86_2/mc/IC86_2_nugen_corrected.npy",
    "aw_path": data_dir + "/DeclinationAcceptance/IC86_2AndFollowing",
    "Livetime": 330.38 + 359.95 + 367.21,
    "Start (MJD)": 56062.420706,
    "End (MJD)": 57160.0440856
}

ps_7year = [IC79_dict, IC86_1_dict, IC86_234_dict]
ps_7year = [IC79_dict]
