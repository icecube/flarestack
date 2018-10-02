import csv
from flarestack.data.icecube.northern_tracks.nt_v002_p01 import *

# The diffuse sample does not have a PS/GFU-style GoodRunList, with start and
# stop times for each good run. It does, however, have a GRL in the form of
# a standard IC-style text file. This script uses the standard GRLs provided
# with the Diffuse sample to create PS/GFU-style numpy arrays.

grl_dir = nt_data_dir + "GRL/"

print grl_dir
print sorted(os.listdir(grl_dir))

sets = [
    (diffuse_IC59, ps_data_dir + "IC59_GRL.npy", grl_dir + "Used.IC59.txt"),
    (diffuse_IC79, ps_data_dir + "IC79b_GRL.npy",
     grl_dir + "IC79_GRL_NewFormat.txt"),
]

for (season_dict, source_grl, text_grl) in sets:

    new = np.load(source_grl)
    new = np.array(new, dtype=new[0].dtype)

    with open(text_grl) as f:

        reader = csv.reader(f, delimiter=' ')

        print len(new),
        grl_ids = [float(x[0]) for x in reader if x[0] != "RunNum"]
        print len(grl_ids)

        print min(grl_ids), max(grl_ids)

        print min(new["run"]), max(new["run"])

        print new.dtype.names

        for y in grl_ids:
            print type(new[new["run"] == y]), len(new[new["run"] == y])

        check = [len(new[new["run"] == y]) for y in grl_ids]

        print max(check), min(check)

        for y in grl_ids:
            print type(new[new["run"] == y])

        rej = np.array([y for y in new if y["run"] not in grl_ids],
                       )
        print len(new)

    print rej

    output_path = diffuse_grl_pathname(season_dict)

    np.save(output_path, new)

    print output_path

