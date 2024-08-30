"""
:mod:`core.preprocess` [subpackage]

Preprocess the raw data up to obtain a format suitable for analysis.

Raw data consists in spiking times of single neurons in different recording sessions.
Final data consists in pseudo-population z-scored firing rates.
Each module is dedicated to one processing step.

Files
-----
Input
    Raw data from each unit in each session.
Output
    Z-scored firing rates of pseudo-populations.

Modules
-------
core.preprocess.firing_rates
    Convert spikes to firing rates.
    Align trials and epochs.
core.preprocess.validate
    Select the valid trials and units to analyze.
core.preprocess.pseudo
    Build pseudo-trials and pseudo-populations.
core.preprocess.zscores
    Compute Z-scores statistics.

See Also
--------
test_core.test_preprocess: Unit tests for this sub-package.
"""
