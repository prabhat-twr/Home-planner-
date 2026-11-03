"""The object catalog the rule engine reasons about (76 entries).

Each entry: id, label, category, Vastu element (pancha bhoota), mount, typical size (w, d, h) in
metres, and the COCO class that a COCO-pretrained YOLO11 maps onto it (``None`` for entries that
need the custom-trained 76-class model). ``typical size`` is used to push floor-contact points from
an object's front edge back to its footprint centre.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    label: str
    category: str
    element: str | None
    mount: str  # floor | wall | ceiling | counter
    size: tuple[float, float, float]
    coco: str | None = None


_E = CatalogEntry
CATALOG: list[CatalogEntry] = [
    # living
    _E("sofa", "sofa", "furniture", "earth", "floor", (2.0, 0.9, 0.85), "couch"),
    _E("armchair", "armchair", "furniture", "earth", "floor", (0.8, 0.8, 0.9), "chair"),
    _E("coffee_table", "coffee table", "furniture", "earth", "floor", (1.1, 0.6, 0.45)),
    _E("tv", "television", "electronics", "fire", "wall", (1.2, 0.1, 0.7), "tv"),
    _E("tv_unit", "tv unit", "furniture", "earth", "floor", (1.6, 0.45, 0.5)),
    _E("bookshelf", "bookshelf", "storage", "earth", "floor", (0.9, 0.35, 1.8)),
    _E("side_table", "side table", "furniture", "earth", "floor", (0.5, 0.5, 0.55)),
    _E("rug", "rug", "decor", "earth", "floor", (2.0, 1.4, 0.01)),
    _E("floor_lamp", "floor lamp", "lighting", "fire", "floor", (0.4, 0.4, 1.6)),
    _E("ceiling_fan", "ceiling fan", "appliance", "air", "ceiling", (1.2, 1.2, 0.4)),
    _E("air_conditioner", "air conditioner", "appliance", "air", "wall", (1.0, 0.25, 0.3)),
    _E("curtains", "curtains", "decor", None, "wall", (1.5, 0.1, 2.2)),
    _E("bench", "bench", "furniture", "earth", "floor", (1.2, 0.4, 0.45), "bench"),
    # bedroom
    _E("bed", "bed", "furniture", "earth", "floor", (1.8, 2.1, 0.6), "bed"),
    _E("wardrobe", "wardrobe", "storage", "earth", "floor", (1.8, 0.6, 2.1)),
    _E("dresser", "dresser", "storage", "earth", "floor", (1.2, 0.5, 0.8)),
    _E("nightstand", "nightstand", "furniture", "earth", "floor", (0.5, 0.4, 0.55)),
    _E("mirror", "mirror", "decor", "water", "wall", (0.6, 0.05, 1.0)),
    _E("study_desk", "study desk", "furniture", "earth", "floor", (1.2, 0.6, 0.75)),
    _E("office_chair", "office chair", "furniture", "earth", "floor", (0.6, 0.6, 1.0)),
    _E("cradle", "cradle", "furniture", "earth", "floor", (1.0, 0.6, 0.9)),
    # kitchen & dining
    _E("stove", "stove / cooktop", "appliance", "fire", "counter", (0.75, 0.5, 0.1)),
    _E("oven", "oven", "appliance", "fire", "floor", (0.6, 0.6, 0.6), "oven"),
    _E("microwave", "microwave", "appliance", "fire", "counter", (0.5, 0.4, 0.3), "microwave"),
    _E("refrigerator", "refrigerator", "appliance", "water", "floor", (0.7, 0.7, 1.8), "refrigerator"),
    _E("sink", "sink", "sanitary", "water", "counter", (0.6, 0.5, 0.2), "sink"),
    _E("dishwasher", "dishwasher", "appliance", "water", "floor", (0.6, 0.6, 0.85)),
    _E("water_purifier", "water purifier", "appliance", "water", "wall", (0.35, 0.2, 0.5)),
    _E("chimney", "kitchen chimney", "appliance", "air", "wall", (0.9, 0.5, 0.6)),
    _E("dining_table", "dining table", "furniture", "earth", "floor", (1.5, 0.9, 0.75), "dining table"),
    _E("dining_chair", "dining chair", "furniture", "earth", "floor", (0.45, 0.5, 0.9)),
    _E("toaster", "toaster", "appliance", "fire", "counter", (0.3, 0.2, 0.2), "toaster"),
    _E("kettle", "kettle", "appliance", "fire", "counter", (0.2, 0.2, 0.25)),
    _E("water_storage", "water storage / pot", "utility", "water", "floor", (0.4, 0.4, 0.5)),
    _E("gas_cylinder", "gas cylinder", "utility", "fire", "floor", (0.35, 0.35, 0.8)),
    # bathroom & utility
    _E("toilet", "toilet", "sanitary", "water", "floor", (0.4, 0.7, 0.8), "toilet"),
    _E("washbasin", "washbasin", "sanitary", "water", "wall", (0.55, 0.45, 0.85)),
    _E("shower", "shower", "sanitary", "water", "wall", (0.9, 0.9, 2.0)),
    _E("bathtub", "bathtub", "sanitary", "water", "floor", (1.7, 0.75, 0.6)),
    _E("water_heater", "water heater / geyser", "appliance", "fire", "wall", (0.4, 0.4, 0.6)),
    _E("washing_machine", "washing machine", "appliance", "water", "floor", (0.6, 0.6, 0.85)),
    _E("dustbin", "dustbin", "utility", None, "floor", (0.35, 0.35, 0.5)),
    _E("broom", "broom / cleaning", "utility", None, "floor", (0.3, 0.1, 1.2)),
    _E("ironing_board", "ironing board", "utility", None, "floor", (1.2, 0.4, 0.9)),
    _E("shoe_rack", "shoe rack", "storage", "earth", "floor", (0.8, 0.35, 0.9)),
    # spiritual
    _E("pooja_mandir", "pooja mandir", "spiritual", "space", "floor", (0.8, 0.5, 1.5)),
    _E("deity_idol", "deity idol", "spiritual", "space", "counter", (0.2, 0.2, 0.3)),
    _E("diya", "diya / oil lamp", "spiritual", "fire", "counter", (0.1, 0.1, 0.1)),
    _E("tulsi_plant", "tulsi plant", "spiritual", "air", "floor", (0.4, 0.4, 0.8)),
    _E("buddha_statue", "buddha statue", "spiritual", "earth", "counter", (0.3, 0.3, 0.5)),
    _E("laughing_buddha", "laughing buddha", "spiritual", "earth", "counter", (0.2, 0.2, 0.25)),
    _E("crystal", "crystal / crystal ball", "spiritual", "earth", "counter", (0.1, 0.1, 0.1)),
    _E("wind_chime", "wind chime", "decor", "air", "ceiling", (0.2, 0.2, 0.6)),
    _E("bagua_mirror", "bagua mirror", "spiritual", "water", "wall", (0.2, 0.02, 0.2)),
    # decor & nature
    _E("potted_plant", "potted plant", "plant", "air", "floor", (0.4, 0.4, 1.0), "potted plant"),
    _E("cactus", "cactus / thorny plant", "plant", "air", "floor", (0.3, 0.3, 0.6)),
    _E("artificial_plant", "artificial plant", "plant", None, "floor", (0.3, 0.3, 0.8)),
    _E("vase", "vase", "decor", "earth", "counter", (0.2, 0.2, 0.35), "vase"),
    _E("clock", "clock", "decor", None, "wall", (0.35, 0.05, 0.35), "clock"),
    _E("wall_art", "painting / wall art", "decor", None, "wall", (0.9, 0.03, 0.6)),
    _E("photo_frame", "photo frame", "decor", None, "wall", (0.3, 0.03, 0.4)),
    _E("aquarium", "aquarium", "decor", "water", "counter", (0.9, 0.4, 0.5)),
    _E("water_fountain", "water fountain", "decor", "water", "floor", (0.5, 0.5, 0.8)),
    _E("candle", "candle", "decor", "fire", "counter", (0.08, 0.08, 0.15)),
    # storage & valuables
    _E("safe", "safe / locker", "storage", "earth", "floor", (0.5, 0.5, 0.7)),
    _E("cabinet", "cabinet", "storage", "earth", "floor", (0.9, 0.45, 1.0)),
    _E("storage_chest", "storage chest", "storage", "earth", "floor", (1.0, 0.5, 0.5)),
    _E("suitcase", "suitcase", "storage", None, "floor", (0.45, 0.3, 0.7), "suitcase"),
    # electronics & electrical
    _E("laptop", "laptop", "electronics", "fire", "counter", (0.35, 0.25, 0.03), "laptop"),
    _E("computer", "computer / monitor", "electronics", "fire", "counter", (0.6, 0.2, 0.45)),
    _E("inverter", "inverter / UPS", "electronics", "fire", "floor", (0.4, 0.3, 0.4)),
    _E("electrical_panel", "electrical panel", "electrical", "fire", "wall", (0.4, 0.15, 0.5)),
    _E("speaker", "speaker", "electronics", "air", "floor", (0.25, 0.25, 0.4)),
    # misc
    _E("staircase", "staircase", "structure", "earth", "floor", (1.0, 3.0, 2.8)),
    _E("gym_equipment", "gym equipment", "furniture", "earth", "floor", (1.5, 0.8, 1.4)),
    _E("books", "books", "decor", None, "counter", (0.3, 0.2, 0.25), "book"),
]

BY_ID = {e.id: e for e in CATALOG}
BY_COCO = {e.coco: e for e in CATALOG if e.coco}
_BY_LABEL = {e.label: e for e in CATALOG}


def resolve(label: str) -> CatalogEntry | None:
    """Map a detector label (catalog id, catalog label, or COCO class name) to a catalog entry."""
    key = label.strip().lower()
    return BY_ID.get(key.replace(" ", "_")) or _BY_LABEL.get(key) or BY_COCO.get(key)
