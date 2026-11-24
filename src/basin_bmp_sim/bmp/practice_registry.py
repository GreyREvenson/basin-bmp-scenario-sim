"""Explicit response-family mapping without hard-coded driver CPS branches.

An unregistered CPS continues to use the historical generic in-field
response; strict practice-specific validation is reserved for a later
scientific-modeling change, not silently introduced during refactoring.
"""
from __future__ import annotations
from basin_bmp_sim.bmp.practice_definition import PracticeDefinition
from basin_bmp_sim.bmp.wetland.constructed_wetland_656 import PRACTICE as CONSTRUCTED_WETLAND
from basin_bmp_sim.bmp.wetland.wetland_restoration_657 import PRACTICE as WETLAND_RESTORATION
from basin_bmp_sim.bmp.grassed_waterway.grassed_waterway_412 import PRACTICE as GRASSED_WATERWAY
from basin_bmp_sim.bmp.infield.cover_crop_340 import PRACTICE as COVER_CROP
from basin_bmp_sim.bmp.infield.no_till_329 import PRACTICE as NO_TILL
from basin_bmp_sim.bmp.infield.nutrient_management_590 import PRACTICE as NUTRIENT_MANAGEMENT

PRACTICES = {p.cps: p for p in (
    CONSTRUCTED_WETLAND, WETLAND_RESTORATION, GRASSED_WATERWAY,
    COVER_CROP, NO_TILL, NUTRIENT_MANAGEMENT,
)}


def resolve_practice(cps: int) -> PracticeDefinition:
    code = int(cps)
    return PRACTICES.get(code, PracticeDefinition(code, f"CPS {code}", "infield"))


def response_family(cps: int) -> str:
    return resolve_practice(cps).response_family
