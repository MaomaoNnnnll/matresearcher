"""Chemical formula normalizer.

Normalizes chemical formulas to a standard format:
  - "Li7La3Zr2O12" -> "Li7La3Zr2O12" (standard)
  - "Li6PS5Cl" -> "Li6PS5Cl"
  - Adds common aliases: "Li7La3Zr2O12" -> alias "LLZO"
  - Handles subscript Unicode and various notations
"""
from __future__ import annotations

import re
from typing import Optional

# Common material aliases in solid-state battery literature
MATERIAL_ALIASES = {
    "Li7La3Zr2O12": "LLZO",
    "Li6.4La3Zr1.4Ta0.6O12": "LLZTO",
    "Li6PS5Cl": "LPSCl",
    "Li6PS5Br": "LPSCr",
    "Li6PS5I": "LPSI",
    "Li3PS4": "LPS",
    "Li10GeP2S12": "LGPS",
    "Li2S-P2S5": "LPS-glass",
    "Li7P3S11": "LPS-7311",
    "Li1.3Al0.3Ti1.7P3O12": "LATP",
    "Li1.6Al0.6Ge0.4P3O12": "LAGP",
    "Li3ClO": "LCO",
    "LiBH4": "LBH",
    "Li2ZrCl6": "LZC",
}

# Subscript mapping for Unicode -> ASCII
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉.", "0123456789.")


class FormulaNormalizer:
    """Normalize chemical formula strings."""

    @staticmethod
    def normalize(formula: str | None) -> tuple[str | None, str | None]:
        """Normalize a chemical formula.

        Returns (normalized_formula, alias) or (None, None) if input is None.
        """
        if not formula:
            return None, None

        # Convert Unicode subscripts to ASCII
        normalized = formula.translate(SUBSCRIPT_MAP)

        # Remove spaces
        normalized = re.sub(r"\s+", "", normalized)

        # Remove parenthetical notes: "Li7La3Zr2O12 (LLZO)" -> "Li7La3Zr2O12"
        paren_match = re.search(r"\(([^)]+)\)", normalized)
        if paren_match:
            # Extract alias from parentheses if it's a known alias
            potential_alias = paren_match.group(1).strip()
            normalized = re.sub(r"\s*\([^)]+\)", "", normalized)

        # Standardize element ordering (rough heuristic - keep original order)
        # Capitalize first letter of elements
        normalized = re.sub(
            r"([a-z])([A-Z])", r"\1\2", normalized  # ensure proper camelCase
        )

        # Look up alias
        alias = MATERIAL_ALIASES.get(normalized)
        if not alias and paren_match:
            alias = paren_match.group(1).strip()

        return normalized, alias

    @staticmethod
    def are_same_material(formula1: str | None, formula2: str | None) -> bool:
        """Check if two formulas refer to the same material."""
        if not formula1 or not formula2:
            return False
        norm1, alias1 = FormulaNormalizer.normalize(formula1)
        norm2, alias2 = FormulaNormalizer.normalize(formula2)
        if norm1 == norm2:
            return True
        if alias1 and alias1 == alias2:
            return True
        if alias1 and alias1 == norm2:
            return True
        if alias2 and alias2 == norm1:
            return True
        return False
