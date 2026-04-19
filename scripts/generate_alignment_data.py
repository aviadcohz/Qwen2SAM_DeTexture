#!/usr/bin/env python3
"""
Generate a large, semantically-constrained text dataset for offline Bridge
alignment (Knowledge Distillation from SAM 3's native text encoder).

KEY PRINCIPLE — GIGO avoidance:
  Descriptions are generated PER-CATEGORY. Each category has its own noun
  inventory, visual-feature adjectives, and color palette. Categories never
  cross: "fluffy" combines with "cat" but never with "concrete"; "rusted"
  combines with "steel" but never with "grass". Shared pieces (spatial
  context, the format template) are the only cross-category elements.

Sources:
  1. ADE20K metadata — real descriptions from our existing training set
     (NOT RWTD). Provides grounded distribution anchor.
  2. Combinatorial synthesis within 15 semantic categories (~500K samples).

Post-generation filter:
  - Drops contradictory adjective pairs (smooth↔rough, wet↔dry, …).
  - Drops grammatically broken or below-minimum-length entries.
  - Deduplicates exactly.

Output format per sample:
    "TEXTURE_1: Texture of <name>, <features>, <context> <|seg|>"

Save path: offline_alignment_dataset.json (list[str]).
RWTD is strictly NOT touched.
"""

import argparse
import itertools
import json
import logging
import random
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gen_alignment")


# ===================================================================== #
#  Category-constrained vocabulary                                        #
# ===================================================================== #

# Each category carries four lists:
#   nouns     — the texture-bearing subject
#   features  — visual / surface adjectives that make sense for this category
#   colors    — colors that are common for this category
#   patterns  — surface patterns that logically belong to this category
#
# Cross-category combinations are prevented by construction.

CATEGORIES = {
    "animals_mammals": {
        "nouns": [
            "cat fur", "dog coat", "rabbit fur", "horse hide", "cow hide",
            "sheep wool", "goat hair", "leopard skin", "tiger fur",
            "zebra hide", "bear fur", "wolf fur", "fox fur", "deer hide",
            "squirrel fur", "elephant skin", "rhino hide", "giraffe hide",
            "cheetah fur", "lion mane",
        ],
        "features": [
            "fluffy fur", "smooth coat", "soft fur", "dense fur",
            "short hair", "long hair", "sleek coat", "rough hide",
            "coarse hair", "silky fur",
        ],
        "colors": [
            "black", "white", "brown", "golden", "tan", "gray",
            "cream-colored", "reddish-brown", "spotted black-and-white",
            "striped orange-and-black",
        ],
        "patterns": [
            "spotted pattern", "striped markings", "solid coloration",
            "mottled pattern", "dappled pattern",
        ],
    },
    "animals_reptiles": {
        "nouns": [
            "snake skin", "lizard scales", "gecko skin", "iguana skin",
            "crocodile hide", "turtle shell", "tortoise shell",
            "chameleon skin",
        ],
        "features": [
            "scaly skin", "smooth scales", "rough scales", "overlapping scales",
            "glossy scales", "dry skin",
        ],
        "colors": [
            "green", "brown", "olive", "yellow-green", "gray-brown",
            "dark green", "mottled green-and-brown",
        ],
        "patterns": [
            "diamond pattern", "geometric scale pattern", "mottled markings",
            "banded pattern", "reticulated pattern",
        ],
    },
    "animals_birds": {
        "nouns": [
            "eagle feathers", "parrot plumage", "peacock feathers",
            "owl feathers", "flamingo plumage", "chicken feathers",
            "duck feathers", "pigeon feathers", "hawk plumage",
        ],
        "features": [
            "feathered plumage", "soft down", "layered feathers",
            "smooth feathers", "ruffled feathers",
        ],
        "colors": [
            "white", "black", "brown", "gray", "iridescent blue",
            "bright red", "yellow", "iridescent green",
        ],
        "patterns": [
            "striped plumage", "speckled pattern", "solid coloration",
            "barred pattern",
        ],
    },
    "animals_marine": {
        "nouns": [
            "fish scales", "shark skin", "octopus skin", "starfish surface",
            "crab shell", "lobster shell", "conch shell", "seashell surface",
            "coral surface", "jellyfish body",
        ],
        "features": [
            "wet scales", "slippery surface", "rough shell", "hard shell",
            "smooth surface", "bumpy surface", "spiny surface",
        ],
        "colors": [
            "silver", "blue-silver", "white", "pink", "red", "coral",
            "brown", "iridescent",
        ],
        "patterns": [
            "mosaic pattern", "spotted pattern", "ridged pattern",
            "spiral pattern",
        ],
    },
    "insects": {
        "nouns": [
            "butterfly wings", "beetle shell", "ladybug shell",
            "dragonfly wings", "spider web", "ant body", "moth wings",
            "bee body", "wasp body", "grasshopper shell", "cricket body",
            "cicada wings", "mantis body",
        ],
        "features": [
            "delicate wings", "hard exoskeleton", "translucent membrane",
            "fine bristles", "glossy shell", "powdery dust",
            "segmented body", "chitinous plates",
        ],
        "colors": [
            "black", "red", "yellow", "iridescent", "metallic green",
            "orange-and-black", "deep blue", "pearly white",
            "bronze-colored", "purplish",
        ],
        "patterns": [
            "spotted pattern", "veined pattern", "symmetrical pattern",
            "banded pattern", "eyespot pattern", "speckled pattern",
        ],
    },
    "vegetation": {
        "nouns": [
            "grass blades", "moss patch", "fern leaves", "ivy vines",
            "bush foliage", "hedge leaves", "flower petals", "tree leaves",
            "vine cluster", "leafy branches", "dense undergrowth",
        ],
        "features": [
            "leafy foliage", "dense foliage", "wispy blades",
            "broad leaves", "thin leaves", "damp foliage", "dry leaves",
        ],
        "colors": [
            "green", "deep green", "yellow-green", "autumn orange",
            "autumn red", "dark green", "brown",
        ],
        "patterns": [
            "veined pattern", "serrated edges", "uniform coverage",
            "clustered growth",
        ],
    },
    "ground_natural": {
        "nouns": [
            "sand", "soil", "dirt", "gravel", "mud", "snow", "ice",
            "pebbles", "dry earth", "forest floor", "beach sand",
            "mountain soil", "desert sand", "volcanic ash", "clay",
            "packed earth", "riverbed mud", "frozen ground",
        ],
        "features": [
            "granular texture", "fine grains", "coarse grains",
            "rough surface", "wet surface", "dry surface",
            "compacted texture", "loose texture", "powdery surface",
            "crusted surface",
        ],
        "colors": [
            "beige", "tan", "brown", "dark brown", "reddish-brown",
            "gray", "white", "golden", "black",
        ],
        "patterns": [
            "rippled pattern", "scattered particles", "uniform grain",
            "footprint-marked", "undisturbed",
        ],
    },
    "rocks_stones": {
        "nouns": [
            "granite", "marble", "limestone", "sandstone", "basalt",
            "quartz", "slate", "shale", "boulder surface", "riverbed stones",
            "stone cliff",
        ],
        "features": [
            "rough surface", "smooth surface", "weathered texture",
            "craggy texture", "cracked surface", "polished finish",
            "jagged edges",
        ],
        "colors": [
            "gray", "dark gray", "white", "beige", "black", "reddish",
            "pinkish", "layered",
        ],
        "patterns": [
            "veined pattern", "speckled pattern", "layered strata",
            "mottled markings", "cracked pattern",
        ],
    },
    "water_fluids": {
        "nouns": [
            "water surface", "ocean waves", "river current", "lake surface",
            "puddle", "pond surface", "flowing stream", "still water",
            "foam", "ripples",
        ],
        "features": [
            "reflective surface", "wavy surface", "still surface",
            "rippled surface", "translucent", "clear", "murky",
        ],
        "colors": [
            "blue", "dark blue", "blue-green", "turquoise", "clear",
            "murky brown", "muddy",
        ],
        "patterns": [
            "rippled pattern", "wave pattern", "smooth reflection",
            "foam-covered",
        ],
    },
    "wood": {
        "nouns": [
            "oak wood", "pine wood", "tree bark", "tree trunk", "driftwood",
            "wooden planks", "plywood", "weathered timber", "log surface",
            "aged wood", "raw lumber",
        ],
        "features": [
            "rough grain", "smooth finish", "knotted texture",
            "weathered surface", "splintered", "polished finish",
            "rustic texture",
        ],
        "colors": [
            "brown", "dark brown", "golden brown", "reddish-brown",
            "weathered gray", "pale beige", "dark stained",
        ],
        "patterns": [
            "grain pattern", "knotted pattern", "cracked pattern",
            "parallel grain",
        ],
    },
    "fabrics": {
        "nouns": [
            "cotton fabric", "wool fabric", "silk", "denim", "linen",
            "velvet", "leather", "fur coat", "corduroy", "canvas",
            "tweed", "satin",
        ],
        "features": [
            "woven pattern", "knit structure", "soft fibers", "ribbed texture",
            "smooth sheen", "coarse weave", "plush pile", "supple surface",
        ],
        "colors": [
            "white", "black", "navy blue", "red", "beige", "gray",
            "dark green", "cream",
        ],
        "patterns": [
            "plaid pattern", "floral pattern", "striped pattern",
            "solid color", "checkered pattern", "herringbone pattern",
        ],
    },
    "metals": {
        "nouns": [
            "steel sheet", "iron plate", "copper surface", "brass surface",
            "aluminum panel", "rusted metal", "corroded steel",
            "galvanized iron", "stainless steel", "bronze", "tin plating",
            "wrought iron", "weathered copper", "polished chrome",
        ],
        "features": [
            "polished surface", "rusted surface", "corroded surface",
            "brushed finish", "dented surface", "scratched surface",
            "reflective finish", "oxidized coating", "tarnished finish",
            "pitted surface",
        ],
        "colors": [
            "silver", "gray", "dark gray", "copper", "golden brass",
            "orange-brown rust", "greenish patina",
        ],
        "patterns": [
            "cross-hatched pattern", "rivet pattern", "scratch marks",
            "oxidation spots", "smooth polish",
        ],
    },
    "architecture": {
        "nouns": [
            "brick wall", "concrete wall", "stone wall", "tiled floor",
            "asphalt road", "pavement", "plaster wall", "cobblestone path",
            "ceramic tiles", "terracotta roof", "marble floor",
        ],
        "features": [
            "weathered surface", "moss-covered", "cracked surface",
            "painted finish", "rough texture", "polished finish",
            "grouted joints",
        ],
        "colors": [
            "gray", "red-brown brick", "beige", "dark gray asphalt",
            "white plaster", "terracotta", "pale cream",
        ],
        "patterns": [
            "brick pattern", "grid tile pattern", "cobblestone pattern",
            "irregular stone pattern", "cracked pattern",
        ],
    },
    "food": {
        "nouns": [
            "bread crust", "cake surface", "cheese surface", "dough",
            "rice grains", "pasta surface", "fruit peel", "chocolate",
            "pizza crust", "biscuit", "cookie surface", "muffin top",
            "pastry layers", "cracker surface", "pretzel surface",
            "sausage skin", "meat surface", "dried fruit",
        ],
        "features": [
            "crumbly surface", "soft texture", "crispy surface",
            "smooth glaze", "flaky layers", "porous texture",
            "glossy coating", "granular surface", "bubbly surface",
        ],
        "colors": [
            "golden brown", "white", "cream", "dark brown", "yellow",
            "reddish", "orange", "pale beige", "honey-colored",
        ],
        "patterns": [
            "crumb pattern", "porous pattern", "smooth icing",
            "glazed surface", "cracked surface", "ridged pattern",
        ],
    },
    "paper_plastic": {
        "nouns": [
            "paper sheet", "cardboard", "plastic surface", "rubber mat",
            "foam sheet", "crumpled paper", "packaging plastic",
            "fabric-like paper", "tissue paper", "parchment paper",
            "bubble wrap", "polyethylene film", "recycled cardboard",
            "cork sheet",
        ],
        "features": [
            "smooth sheet", "crumpled surface", "ribbed surface",
            "porous surface", "glossy finish", "matte finish",
            "textured surface", "perforated surface", "fibrous surface",
        ],
        "colors": [
            "white", "beige", "gray", "black", "yellow", "transparent",
            "kraft brown", "pale cream", "recycled gray",
        ],
        "patterns": [
            "corrugated pattern", "smooth sheet", "ribbed lines",
            "printed pattern", "crumple pattern", "honeycomb pattern",
        ],
    },
}

# Shared across all categories.
SPATIAL_CONTEXTS = [
    "foreground", "background", "center", "top-left", "top-right",
    "bottom-left", "bottom-right", "left side", "right side",
    "upper half", "lower half", "central area",
    "covering the entire surface", "occupying the main area",
]


# ===================================================================== #
#  Contradiction / sanity rules                                           #
# ===================================================================== #

# Pairs of adjectives that must NOT co-occur in a single description.
# These are checked on tokenised lowercase text (word set).
CONTRADICTIONS = [
    {"smooth", "rough"},
    {"wet", "dry"},
    {"soft", "hard"},
    {"fluffy", "metallic"},
    {"shiny", "matte"},
    {"hot", "cold"},
    {"liquid", "solid"},
    {"translucent", "opaque"},
    {"dense", "sparse"},
    {"coarse", "fine"},
    {"polished", "weathered"},
]

MIN_WORDS = 8       # post-template minimum word count
MAX_WORDS = 30      # reject rambling entries (also as a safety valve)


_TEXTURE_N_RE = re.compile(r"texture_\d+", re.IGNORECASE)


def passes_sanity(text: str) -> bool:
    """Return True iff the description is logically and grammatically plausible."""
    # Minimum length
    word_list = re.findall(r"\b[\w'-]+\b", text.lower())
    if not (MIN_WORDS <= len(word_list) <= MAX_WORDS):
        return False
    words = set(word_list)

    # Contradiction pairs
    for pair in CONTRADICTIONS:
        if pair.issubset(words):
            return False

    # Must contain the expected skeleton tokens
    if not _TEXTURE_N_RE.search(text) or "<|seg|>" not in text:
        return False
    if ": texture of" not in text.lower():
        return False

    # No unbalanced characters
    if text.count(",") < 2:
        # our template has at least 2 commas: after name, after features
        return False

    return True


# ===================================================================== #
#  Generators                                                             #
# ===================================================================== #

TEMPLATE = "TEXTURE_{n}: Texture of {name}, {features}, {context} <|seg|>"

# During real V7 training a sample has 1..6 textures, so Qwen sees TEXTURE_N
# prefixes where N is any of 1..6 (most samples are k=1-3 in RWTD). We
# enumerate all six to make the Bridge prefix-invariant.
TEXTURE_INDICES = [1, 2, 3, 4, 5, 6]


def format_description(name: str, features: str, context: str, n: int) -> str:
    return TEMPLATE.format(n=n, name=name, features=features, context=context)


def extract_ade20k_descriptions(metadata_path: Path) -> list[str]:
    """Pull every GT description from ADE20K metadata and reformat to our
    template. Provides a grounded anchor in the real training distribution.
    Returns deduplicated list of fully-formatted strings."""
    if not metadata_path.exists():
        log.warning(f"ADE20K metadata not found at {metadata_path} — skipping.")
        return []

    with open(metadata_path) as f:
        meta = json.load(f)

    out = set()
    for entry in meta:
        for tex in entry.get("textures", []):
            desc = tex.get("description", "").strip()
            if not desc:
                continue
            # ADE20K descriptions are already like "Texture of X, Y, Z".
            # Wrap with TEXTURE_N: prefix (N varies 1..6 to match V7 training
            # distribution) + <|seg|> suffix.
            if desc.lower().startswith("texture of"):
                core = desc[len("texture of"):].lstrip(", ").strip()
                body = f"Texture of {core}"
            else:
                body = f"Texture of {desc}"
            for n in TEXTURE_INDICES:
                out.add(f"TEXTURE_{n}: {body} <|seg|>")
    log.info(f"Loaded {len(out)} unique ADE20K descriptions (across TEXTURE_N).")
    return list(out)


def generate_synthetic_for_category(cat: dict, target_count: int,
                                    rng: random.Random) -> list[str]:
    """Exhaustively enumerate the valid combinatorial space for one category
    (including TEXTURE_N prefix variation over N ∈ {1..6} to prevent the
    Bridge from learning an index bias), apply sanity filter, deduplicate,
    shuffle, and return up to `target_count` descriptions. No cross-category
    contamination.

    For each (noun, color, pattern, feature, spatial, N) tuple we emit TWO
    grammatical framings:
        "<color> <pattern>"            (pattern-led)
        "<color> with <feature>"       (feature-led)
    Max unique per category = |nouns| × |colors| × |patterns| × |features|
                              × |spatials| × |N-values| × 2.
    """
    combos = []
    for name, color, pattern, feat, spatial, n in itertools.product(
        cat["nouns"], cat["colors"], cat["patterns"], cat["features"],
        SPATIAL_CONTEXTS, TEXTURE_INDICES,
    ):
        combos.append(format_description(name, f"{color} {pattern}", spatial, n))
        combos.append(format_description(name, f"{color} with {feat}", spatial, n))

    sane = [c for c in combos if passes_sanity(c)]
    unique = list(set(sane))
    rng.shuffle(unique)
    if len(unique) > target_count:
        unique = unique[:target_count]
    return unique


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ade20k-metadata",
                        default="/home/aviad/datasets/ADE20K_textured_images/train_metadata.json",
                        help="Path to ADE20K training metadata (NOT RWTD).")
    parser.add_argument("--n-per-category", type=int, default=35000,
                        help="Target synthetic samples per category.")
    parser.add_argument("--output",
                        default=str(_PROJECT_ROOT / "offline_alignment_dataset.json"),
                        help="Path for the generated JSON dataset.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    log.info("Step 1 — harvest real ADE20K descriptions (non-RWTD).")
    real = extract_ade20k_descriptions(Path(args.ade20k_metadata))
    real_filtered = [t for t in real if passes_sanity(t)]
    log.info(f"  Real: {len(real)} raw, {len(real_filtered)} passed sanity.")

    log.info("Step 2 — synthesize per-category constrained combinations.")
    synth_all = []
    for cat_name, cat in CATEGORIES.items():
        n_cat = args.n_per_category
        log.info(f"  [{cat_name}] target {n_cat}...")
        syn = generate_synthetic_for_category(cat, n_cat, rng)
        log.info(f"  [{cat_name}] generated {len(syn)} unique+sane.")
        synth_all.extend(syn)

    log.info("Step 3 — merge, deduplicate, shuffle.")
    merged = set(real_filtered)
    pre = len(merged)
    merged.update(synth_all)
    log.info(f"  Pre-synth: {pre}   Post-synth: {len(merged)}   "
             f"Unique synth added: {len(merged) - pre}")

    final = list(merged)
    rng.shuffle(final)

    log.info("Step 4 — final sanity sweep.")
    final_filtered = [t for t in final if passes_sanity(t)]
    dropped = len(final) - len(final_filtered)
    log.info(f"  Dropped {dropped} in final sweep (should normally be 0 "
             f"because filters were applied upstream).")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Step 5 — saving {len(final_filtered)} entries to {out_path}.")
    with open(out_path, "w") as f:
        json.dump(final_filtered, f, indent=2, ensure_ascii=False)

    log.info("Done.")
    log.info(f"Example [0]: {final_filtered[0]}")
    log.info(f"Example [n/2]: {final_filtered[len(final_filtered) // 2]}")
    log.info(f"Example [-1]: {final_filtered[-1]}")


if __name__ == "__main__":
    main()
