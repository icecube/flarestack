# Data types
The module `core.data_types` is meant to collect different data types used by the different modules of *flarestack*, so they can be inspected and manipulated by the user in an independent fashion.

Currently, it only includes `catalogue_dtype`.

## Catalogue
A *flarestack* catalogue takes the form of a `numpy` [structured array](https://numpy.org/doc/stable/user/basics.rec.html), consisting of the following fields:
- `ra_rad` (`np.float`): right ascension (J2000.0) of the source in radians;
- `dec_rad` (`np.float`): declination of the source in radians;
- `base_weight` (`np.float`): base weight of the source for injection and fitting. Base weights should **not** include the distance scaling (this is handled implicitly by *flarestack*). *flarestack* takes care of normalising base weights to their sum, so the absolute scale of `base_weight` is not important.
- `injection_weight_modifier` (`np.float`): multiplicative factor for `base _weight` only used in the signal injection. By default it should be set to one. Proper usage implies taking care of preserving the overall flux normalisation, so handle with care. 
- `ref_time_mjd` (`np.float`): reference time, modified Julian day.
- `start_time_mjd` (`np.float`): start time for the source time window, modifed Julian day. This is used for time-dependent injection and fitting.
- `end_time_mjd` (`np.float`): end time for the source time window, modified Julian day. This is used for time-dependent injection and fitting.
- `distance_mpc` (`np.float`): distance of the source, in Mpc (megaparsecs).
- `source_name` (string): name of the source.
