**Script Descriptions**

These scripts use the background TS distributions created by ```decay_sensitivity.py``` and ```box_sensitivity.py```,
which have to be run first!

* ```unblind_update.py```: Calculate the pre-trial p-values from the actual IceCube data
* ```p_value_distribution/generate_background_pvalue_distributions.py```: Generate p-value distributions from simulated background 
datasets using ```p_value_distribution/generate_background_pvalue_distribution_single.py``` to trial correct the pre-trial p-values
* ```energy_range/energy_range.py```: Run trials for the energy range calculation. 
* ```energy_range/energy_range.ipynb```: Calculate the energy range that this analysis is sensitive to, 
```energy_range.py``` has to be run first!
* ```unblinding_plots.ipynb```: visualize the results of ```unblind_updated_ccsn.py```, calculate post_trial p-values, 
calculate fluences and upper limits