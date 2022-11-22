import numpy as np
import os
import logging
from astropy import units as u
from flarestack.cosmo.rates.ccsn_rates import get_ccsn_rate


dir_path = os.path.dirname(os.path.realpath(__file__))


def get_ccsn_percentage(sn_type):

    with open(f"{dir_path}/ccsn_subtypes_percentage.tsv", "r") as f:
        raw = f.read()

    r = [i.split("\n") for i in raw.split("\t")]
    o = list()
    for i in r:
        for j in i:
            if "I" in j or "%" in j or "Faint CCSN" in j:
                o.append(j)
    types = np.array(o)[np.linspace(0, len(o) / 2 - 1, int(len(o) / 2), dtype=int) * 2]
    percentages = np.array(o)[
        np.linspace(0, len(o) / 2 - 1, int(len(o) / 2), dtype=int) * 2 + 1
    ]

    perc = float(percentages[types == sn_type][0].strip("%")) / 100

    return perc


def get_ccsn_rate_function(source):

    if source == "candles_clash":

        file_path = f"{dir_path}/candles_clash_ccsn_rate.tsv"
        logging.debug(f"loading {file_path}")
        with open(f"{dir_path}/candles_clash_ccsn_rate.tsv", "r") as f:
            raw = f.read()

        r = [i.replace("\n", "") for i in raw.split("\t")]

        z_err = 0.2
        z = np.array([float(i.split(" +or-")[0]) for i in r if "+or-" in i])

        candles_and_clash_indices = (
            np.linspace(0, len(z) - 1, len(z), dtype=int) * 3 + 2
        )

        rate_candles_and_clash = np.array(
            [
                float(i.split("{")[1].split("}")[0])
                for i in r
                if "${" in i and "h" not in i
            ]
        )[candles_and_clash_indices]

        rate_candles_and_clash_e = np.array(
            [
                (
                    float(i.split("{-")[1].split("}")[0]),
                    float(i.split("{")[3].split("}")[0]),
                )
                for i in r
                if "${" in i and "h" not in i
            ]
        )[candles_and_clash_indices]

        def rate(redshift, H_0):

            if redshift < z[1] - z_err:
                logging.warning("using step interpolation for rate!")
                res = rate_candles_and_clash[0]
            elif redshift > z[-1] + z_err:
                logging.warning(
                    "assuming rate stays constant after largest available value for redshift!"
                )
                res = rate_candles_and_clash[-1]
            else:
                res = rate_candles_and_clash[
                    (redshift >= z - z_err) & (redshift <= z + z_err)
                ]

            return float(res) * (H_0 / 70) ** 3 * 1e-4 * u.Mpc ** (-3) * u.yr ** (-1)

    elif source == "candles_clash_plus_literatur":

        file_path = f"{dir_path}/candles_clash_plus_literatur_ccsn_rate.tsv"
        logging.debug(f"loading {file_path}")
        with open(file_path, "r") as f:
            raw = f.read()

        r = np.array([i for i in raw.split("\n") if i])
        o = r[int(np.where(r == "^c:")[0] + 1) : int(np.where(r == "Notes.")[0])]

        z = np.array(
            [
                (
                    float(i.split("$")[1].split("\\pm ")[0]),
                    float(i.split("$")[1].split("\\pm ")[1]),
                )
                for i in o[
                    np.linspace(0, len(o) / 3 - 1, int(len(o) / 3), dtype=int) * 3
                ]
            ],
            dtype={"names": ["z", "z_err"], "formats": ["<f8"] * 2},
        )

        rate_candles_and_clash_and_literatur = np.array(
            [
                float(i.split("${")[1].split("}")[0])
                for i in o[
                    np.linspace(0, len(o) / 3 - 1, int(len(o) / 3), dtype=int) * 3 + 1
                ]
            ]
        )

        def rate(redshift, H_0):
            if redshift < z["z"][1] - z["z_err"][1]:
                logging.warning("using step interpolation for rate!")
                res = rate_candles_and_clash_and_literatur[0]
            elif redshift > z["z"][-1] + z["z_err"][-1]:
                logging.warning(
                    "assuming rate stays constant after largest available value for redshift!"
                )
                res = rate_candles_and_clash_and_literatur[-1]
            else:
                res = rate_candles_and_clash_and_literatur[
                    (redshift >= z["z"] - z["z_err"])
                    & (redshift <= z["z"] + z["z_err"])
                ][0]
            return float(res) * (H_0 / 70) ** 3 * 1e-4 * u.Mpc**-3 * u.yr**-1

    else:
        raise ValueError(f"Sources {source} not known!")

    return rate


def _get_ccsn_rate_type_function(
    sn_type, source="candles_clash_plus_literatur", H_0=70
):

    rate_function = get_ccsn_rate_function(source)
    type_percentage = (
        get_ccsn_percentage(sn_type)
        if sn_type != "Ibc"
        else get_ccsn_percentage("Ib") + get_ccsn_percentage("Ic")
    )

    def ccsn_type_rate(redshift):
        return rate_function(redshift, H_0) * type_percentage

    return ccsn_type_rate


def get_ccsn_rate_type_function(sn_type):
    source = "strolger_15"
    return get_ccsn_rate(source, source, source, sn_type)
