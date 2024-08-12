#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_objects.test_composites` [module]

Tests for the module :mod:`core.core_objects.composites`.
"""

import pytest

from core.entities.composites import Site, Unit, Session


def test_split_id_unit():
    """
    Test :meth:`Unit.split_id`.
    """
    unit = Unit("avo052a-d1")
    assert unit.split_id() == ("avo052a", "avo", "a")


def test_split_id_session():
    """
    Test :meth:`Session.split_id`.
    """
    session = Session("avo052a04_p_PTD")
    assert session.split_id() == ("avo052a", 4, "p", "PTD")


def test_bidirectional_association():
    """
    Test :meth:`Site.add_unit` and :meth:`Unit.set_site`.

    As the site is set automatically when a unit is instantiated,
    it is only necessary to test
    - Whether the site is correctly set for this unit.
    - Whether the unit is recorded among the units of the site.
    """
    site = Site("avo052a")
    unit = Unit("avo052a-d1")
    assert unit.site == site
    assert unit in site.units


def test_site_init():
    """
    Test :meth:`Site.__init__`.
    """
    site = Site("avo052a")
    assert site.id == "avo052a"


def test_unit_init():
    """
    Test :meth:`Unit.__init__`.
    """
    unit = Unit("avo052a-d1")
    assert unit.id == "avo052a-d1"
    assert unit.site.id == "avo052a"


def test_session_init():
    """
    Test :meth:`Session.__init__`.
    """
    session = Session("avo052a04_p_PTD")
    a = 0
    assert session.id == "avo052a04_p_PTD"
    assert session.site.id == "avo052a"
    assert session.task.value == "PTD"
    assert session.ctx.value == "p"
