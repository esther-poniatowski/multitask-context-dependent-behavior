"""
:mod:`mtcdb.preprocess` [subpackage]

Preprocess the raw data up to obtain a format suitable for analysis.

Raw data consists in spiking times of single neurons in different recording sessions.
Final data consists in pseudo-population zscored firing rates.
Each module is dedicated to one processing step.

Files
-----
Input
    Raw data from each unit in each session.
Output
    Z-scored firing rates of pseudo-populations.

Modules
-------
mtcdb.preprocess.firing_rates
    Convert spikes to firing rates.
    Align trials and epochs.
mtcdb.preprocess.validate
    Select the valid trials and units to analyze.
mtcdb.preprocess.pseudo
    Build pseudo-trials and pseudo-populations.
mtcdb.preprocess.zscores
    Compute Z-scores statistics.

See Also
--------
test_mtcdb.test_preprocess: Unit tests for this sub-package.
"""