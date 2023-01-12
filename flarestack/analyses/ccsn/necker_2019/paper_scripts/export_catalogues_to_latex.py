from flarestack.analyses.ccsn.necker_2019.ccsn_helpers import (
    updated_sn_catalogue_name,
    sn_cats,
)
from flarestack.shared import plot_output_dir
import numpy as np
import logging
import json
import pandas as pd
import os
from astropy.time import Time


logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


name_root = "analyses/ccsn/necker_2019/catalogues"
references_fn = os.path.join(os.path.dirname(__file__), "references.json")


if __name__ == '__main__':

    logging.basicConfig()

    with open(references_fn, "r") as f:
        references = json.load(f)

    # build latex tables

    for cat_name in sn_cats:
        fn = updated_sn_catalogue_name(cat_name)
        cat = pd.DataFrame(np.load(fn)).sort_values("ref_time_mjd")

        # set up dictionary to build reference map
        used_refs = {"": 0}

        txt = ""
        for i, r in cat.iterrows():

            # get the reference number
            refs = references[r.source_name]["references"]
            cite_strings = [f"\citet{{{ref}}}" for ref in refs]
            cite_string = ", ".join(cite_strings)
            ref_numbers = list()
            for ref in refs:

                # if reference is not in the reference map, make the entry
                if not ref in used_refs:
                    used_refs[ref] = max(used_refs.values()) + 1
                ref_numbers.append(used_refs[ref])

            show_numbers = ", ".join(np.array(ref_numbers).astype(str))
            sn_name = references[r.source_name].get("show_name", r.source_name)

            txt += (
                f"{sn_name:30} & "
                f"{r.ra_rad:5.2f} & "
                f"{r.dec_rad:5.2f} & "
                f"{Time(r.ref_time_mjd, format='mjd').strftime('%Y-%M-%d')} & "
                f"{r.redshift:5.4f} & "
                f"{r.distance_mpc:5.2f} & "
                f"{show_numbers}"
                f"\\\\\n"
            )

        # write latex tables

        latex_fn = os.path.join(plot_output_dir(name_root), f"{cat_name}_catalogue.tex")
        logger.info(f"saving {cat_name} under {latex_fn}")

        d = os.path.dirname(latex_fn)
        if not os.path.isdir(d):
            os.makedirs(d)

        with open(latex_fn, "w") as f:
            f.write(txt)

        # write reference map

        refmap_fn = os.path.join(plot_output_dir(name_root), f"{cat_name}_reference_map.tex")
        logger.info(f"saving reference map under {refmap_fn}")

        refmap = [f"({refnum}) \\citet{{{ref}}}" for ref, refnum in used_refs.items() if refnum > 0]
        refmap_txt = ", ".join(refmap)
        with open(refmap_fn, "w") as f:
            f.write(refmap_txt)
