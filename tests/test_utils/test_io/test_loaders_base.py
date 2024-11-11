#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`test_core.test_io_data.test_loaders_base` [module]

Notes
-----
Although those tests aim to cover the method defined in the abstract base class, the concrete
LoaderCSV implementation subclass is used. It is chosen for simplicity and because it offers more
options to test.

Those tests do not check for the concents of the data loaded, but rather for the correct handling of
the file paths and formats.

See Also
--------
`utils.io_data.base_loader`: Tested module.
`utils.io_data.loaders`: Concrete implementations.
"""

import pytest

# from utils.io_data.loaders import LoaderCSVtoDataFrame
