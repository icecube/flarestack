import numpy as np
from flarestack.shared import catalogue_dir
from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1, ps_7year
from flarestack.utils.custom_seasons import custom_dataset
from flarestack.utils.neutrino_astronomy import calculate_neutrinos
from astropy import units as u

# Calculate expected neutrino numbers for entries in the catalogue by Dai &
# Fang, in which nearby TDEs were selected and event rates estimated. The
# numbers they got seem a little 'approximate', but still comparable to the
# numbers from this script. The catalogue and event rates can be found in
# the paper at https://arxiv.org/pdf/1612.00011

cat_path = catalogue_dir + "TDEs/Dai_Fang_TDE_catalogue.npy"

sources = np.load(cat_path)

# In column one of the paper, they assume a base cosmic ray flux of 10^51 erg

cr_flux = 10 ** 51 * u.erg

# They use the standard Waxmann Bachall factor, in which charged pions are
# produced with 50% probability, and 3/4 of the decay products of charged
# pions are neutrinos, giving a factor of 3/8 of the energy of pions being
# transferred to neutrinos

waxmann_bachall = 0.5 * 0.75

# A pion production fraction of 10% is assumed, based on jet/choked jet model.

f_pi = 0.1

# This gives 3.5% conversion of CR energy to neutrino energy

f_nu_to_cr = f_pi * waxmann_bachall

# The Cosmic Ray Energy itself can be calculated assuming an E^-2 power law
# The minimal energy of the power law is 1GeV, and it extends to 10^17 eV.

cr_integral = np.log(10**21/10**9)

int_cr_energy = cr_flux * cr_integral

# nu_energy = cr_energy * f_nu_to_cr

# In the paper, they assume a base cosmic ray energy of 10^51 erg. This is
# inversely propotional to the pion fraction, because it scales the cr flux to
# explain the IceCube neutrino flux.

base_cr_energy = 10**51 * u.erg / f_pi

# They also assume a scaled CR energy of 10x radiated energy.
# These values are copied from the table (and I think some might be wrong...)

radiated_energy = {
    "UGC 03317": 4 * 10**49 * u.erg,
    "PGC 1185375": 10**50 * u.erg,
    "PGC 1190358": 2 * 10**50 * u.erg,
    "PGC 015259": 3 * 10**50 * u.erg,
    "iPTF16fnl": 2 * 10**49 * u.erg,
    "XMMSL1 J0740-85": 5 * 10**50 * u.erg,
    "ASASSN-15oi": 5 * 10**50 * u.erg,
    "ASASSN-14li": 7 * 10**51 * u.erg,
    "ASASSN-14ae": 1.7 * 10 ** 51 * u.erg,
    "Swift J1644+57": 10**53 * 20 * u.erg
}

injection_window = 100

time_pdf = {
    "Name": "Box",
    "Pre-Window": 0,
    "Post-Window": injection_window
}


res_dict = dict()
#
for source in sources:

    # Second table assumes CR energy is 10 times the radiated energy

    custom_cr_energy = radiated_energy[source["Name"]] * 10

    n_injs = []

    print source["Name"]

    for cr_energy in [base_cr_energy, custom_cr_energy]:

        print cr_energy

        time = injection_window * 60 * 60 * 24

        # Convert cosmic ray energy back to differential flux

        cr_flux = cr_energy/(cr_integral * time)

        nu_flux = f_nu_to_cr * cr_flux

        energy_pdf = {
            "Name": "Power Law",
            "Gamma": 2.0,
            "Energy Flux": nu_flux,
            "E Min": 10 ** 2
        }

        inj_kwargs = {
            "Injection Energy PDF": energy_pdf,
            "Injection Time PDF": time_pdf,
            "Poisson Smear?": True,
        }

        seasons = custom_dataset(txs_sample_v1, [source], time_pdf)

        n_inj = 0

        for season in seasons:
            n_inj += calculate_neutrinos(source, season, inj_kwargs)

        n_injs.append('{0:.2f}'.format(n_inj))

    res_dict[source["Name"]] = n_injs

print "\n"

print "N_base \t N_scale \t Source"

for (source, n_inj) in res_dict.iteritems():

    print n_inj[0], "\t", n_inj[1], "\t", source
