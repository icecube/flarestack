"""Script to reproduce the catalogues used in Alexander Stasik's CCSN
stacking analysis. Raw files were provided by him, following the merger of
several supernova catalogues.

The analysis created two subcatalogues for each class. One, a nearby sample,
contained 70% of the signal weight. Another larger sample contained the
remaining 30% weight distributed across many sources. As integer numbers of
sources are used, the closest percentage to 70% is used for splitting.
"""
from flarestack.analyses.ccsn.shared_ccsn import raw_cat_dir, sn_catalogue_name
from flarestack.utils.prepare_catalogue import cat_dtype
import numpy as np
import os

mask = ["ra", "dec", "distance", "discoverydate_mjd", "name", 'weight']
new_names = ["ra", "dec", "Distance (Mpc)", "Ref Date (MJD)", "Name", "weight"]

raw_cats = [x for x in os.listdir(raw_cat_dir) if x[0] != "."]

threshold = 0.7

for sn_cat in raw_cats:

    print "Converting", sn_cat

    cat = np.load(raw_cat_dir + sn_cat)

    sn_type = sn_cat.split("_")[0]

    if sn_type == "Ib":
        sn_type = "Ibc"
    elif sn_type == "IIp":
        sn_type = "IIP"

    cat = cat[mask]
    cat = np.array(sorted(cat, key=lambda x: x["weight"], reverse=True))

    j = 1

    while np.sum((cat["weight"])[:j]) < threshold:
        j += 1

    w_above = np.sum((cat["weight"])[:j])
    w_below = np.sum((cat["weight"])[:j-1])

    # Check whether adding the last source brings the weght closer to
    # threshold or not

    if abs(threshold - w_below) < abs(w_above - threshold):
        j -= 1

    close = cat[:j]
    far = cat[j:]

    for i, subcat in enumerate([close, far]):

        new_cat = np.empty(len(subcat), dtype=cat_dtype)
        new_cat["ra"] = subcat["ra"]
        new_cat["dec"] = subcat["dec"]
        new_cat["Distance (Mpc)"] = subcat["distance"]
        new_cat["Ref Time (MJD)"] = subcat["discoverydate_mjd"]
        new_cat["Name"] = subcat["name"]
        new_cat["Relative Injection Weight"] = np.ones_like(subcat["weight"])

        save_path = sn_catalogue_name(sn_type, nearby=[True, False][i])

        print "Saving to", save_path

        np.save(save_path, new_cat)
