#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_structures.base_data_component` [module]

Classes
-------
DataComponent (base class)

"""
from abc import ABC
import copy
from typing import Tuple, Mapping, Type, Self

from core.data_components.core_data import CoreData, Dimensions

from core.coordinates.base_coord import Coordinate


class DataComponent(ABC):
    """
    Base class for data components.
    """
