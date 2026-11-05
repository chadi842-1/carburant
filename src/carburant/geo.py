"""Reference table of French départements (synthetic source and dim_localisation).

Coordinates are approximate city centres (WGS84). `price_effect` is a SYNTHETIC regional price
offset in EUR/l used only by the generator; it is not a claim about real regional prices.
"""
from __future__ import annotations

DEPARTEMENTS: list[dict] = [
    # code, name, region, main city, lat, lon, synthetic price effect
    {"dep": "75", "nom": "Paris", "region": "Île-de-France", "ville": "Paris",
     "lat": 48.857, "lon": 2.352, "price_effect": 0.06},
    {"dep": "92", "nom": "Hauts-de-Seine", "region": "Île-de-France", "ville": "Nanterre",
     "lat": 48.892, "lon": 2.207, "price_effect": 0.04},
    {"dep": "69", "nom": "Rhône", "region": "Auvergne-Rhône-Alpes", "ville": "Lyon",
     "lat": 45.764, "lon": 4.836, "price_effect": 0.01},
    {"dep": "38", "nom": "Isère", "region": "Auvergne-Rhône-Alpes", "ville": "Grenoble",
     "lat": 45.188, "lon": 5.724, "price_effect": 0.00},
    {"dep": "63", "nom": "Puy-de-Dôme", "region": "Auvergne-Rhône-Alpes",
     "ville": "Clermont-Ferrand", "lat": 45.778, "lon": 3.087, "price_effect": -0.01},
    {"dep": "13", "nom": "Bouches-du-Rhône", "region": "Provence-Alpes-Côte d'Azur",
     "ville": "Marseille", "lat": 43.296, "lon": 5.370, "price_effect": 0.02},
    {"dep": "06", "nom": "Alpes-Maritimes", "region": "Provence-Alpes-Côte d'Azur",
     "ville": "Nice", "lat": 43.710, "lon": 7.262, "price_effect": 0.05},
    {"dep": "59", "nom": "Nord", "region": "Hauts-de-France", "ville": "Lille",
     "lat": 50.629, "lon": 3.057, "price_effect": -0.02},
    {"dep": "31", "nom": "Haute-Garonne", "region": "Occitanie", "ville": "Toulouse",
     "lat": 43.605, "lon": 1.444, "price_effect": -0.01},
    {"dep": "34", "nom": "Hérault", "region": "Occitanie", "ville": "Montpellier",
     "lat": 43.611, "lon": 3.877, "price_effect": 0.00},
    {"dep": "33", "nom": "Gironde", "region": "Nouvelle-Aquitaine", "ville": "Bordeaux",
     "lat": 44.838, "lon": -0.579, "price_effect": 0.00},
    {"dep": "87", "nom": "Haute-Vienne", "region": "Nouvelle-Aquitaine", "ville": "Limoges",
     "lat": 45.834, "lon": 1.261, "price_effect": -0.03},
    {"dep": "44", "nom": "Loire-Atlantique", "region": "Pays de la Loire", "ville": "Nantes",
     "lat": 47.218, "lon": -1.554, "price_effect": -0.02},
    {"dep": "35", "nom": "Ille-et-Vilaine", "region": "Bretagne", "ville": "Rennes",
     "lat": 48.117, "lon": -1.678, "price_effect": -0.03},
    {"dep": "29", "nom": "Finistère", "region": "Bretagne", "ville": "Brest",
     "lat": 48.390, "lon": -4.486, "price_effect": -0.02},
    {"dep": "67", "nom": "Bas-Rhin", "region": "Grand Est", "ville": "Strasbourg",
     "lat": 48.573, "lon": 7.752, "price_effect": 0.00},
    {"dep": "51", "nom": "Marne", "region": "Grand Est", "ville": "Reims",
     "lat": 49.258, "lon": 4.032, "price_effect": -0.01},
    {"dep": "21", "nom": "Côte-d'Or", "region": "Bourgogne-Franche-Comté", "ville": "Dijon",
     "lat": 47.322, "lon": 5.041, "price_effect": 0.00},
    {"dep": "76", "nom": "Seine-Maritime", "region": "Normandie", "ville": "Rouen",
     "lat": 49.443, "lon": 1.099, "price_effect": -0.01},
    {"dep": "45", "nom": "Loiret", "region": "Centre-Val de Loire", "ville": "Orléans",
     "lat": 47.903, "lon": 1.909, "price_effect": -0.01},
]
DEP_BY_CODE = {d["dep"]: d for d in DEPARTEMENTS}


def departement_from_cp(cp: str | None) -> str | None:
    """Département code from a 5-digit postal code (handles Corsica 20xxx -> 2A/2B)."""
    if not cp or len(cp) != 5 or not cp.isdigit():
        return None
    if cp.startswith("20"):
        return "2A" if int(cp) < 20200 else "2B"
    if cp.startswith("97"):
        return cp[:3]
    return cp[:2]
