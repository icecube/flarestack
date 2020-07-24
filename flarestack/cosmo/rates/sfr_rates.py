from astropy import units as u

def sfr_madau(z):
    """
    star formation history
    http://arxiv.org/pdf/1403.0007v3.pdf
    Madau & Dickinson 2014, Equation 15
    result is in solar masses/year/Mpc^3

    """
    rate = 0.015 * (1+z)**2.7 / (1 + ((1+z)/2.9)**5.6) /(
        u.Mpc**3 * u.year
    )

    return rate


def sfr_clash_candels(z):
    """
    star formation history
    https://arxiv.org/pdf/1509.06574.pdf

    result is in solar masses/year/Mpc^3

    Can match Figure 6, if the h^3 factor is divided out
    """
    rate = 0.015 * (1+z)**5.0 / (1 + ((1+z)/1.5)**6.1) /(
        u.Mpc**3 * u.year
    )

    return rate