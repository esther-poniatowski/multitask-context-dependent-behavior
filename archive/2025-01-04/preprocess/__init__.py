"""
:mod:`core.processors.preprocess` [subpackage]

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
core.processors.preprocess.firing_rates
    Convert spikes to firing rates.
    Align trials and epochs.
core.processors.preprocess.validate
    Select the valid trials and units to analyze.
core.processors.preprocess.pseudo
    Build pseudo-trials and pseudo-populations.
core.processors.preprocess.zscores
    Compute Z-scores statistics.

See Also
--------
test_core.test_preprocess: Unit tests for this sub-package.
"""
