import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class CleaningStats:
    removed_periods: int = 0
    normalized_whitespace: int = 0
    fixed_capitalization: int = 0

    @classmethod
    def from_counts(cls, counts: Dict[str, int]) -> "CleaningStats":
        return cls(
            removed_periods=counts.get("removed_periods", 0),
            normalized_whitespace=counts.get("normalized_whitespace", 0),
            fixed_capitalization=counts.get("fixed_capitalization", 0),
        )

    def merge(self, other: "CleaningStats") -> None:
        self.removed_periods += other.removed_periods
        self.normalized_whitespace += other.normalized_whitespace
        self.fixed_capitalization += other.fixed_capitalization

    def to_dict(self) -> Dict[str, int]:
        return {
            "removed_periods": self.removed_periods,
            "normalized_whitespace": self.normalized_whitespace,
            "fixed_capitalization": self.fixed_capitalization,
        }


class MathBridgeCleaner:
    def __init__(self):
        # precompiled regexes
        self._multi_space = re.compile(r"\s+")
        # Periods erroneously at end of spoken text lines (keep ellipses ...).
        self._end_period = re.compile(r"(?<!\.)\.(?=\s*$)")

    def _clean_spoken(self, text: str) -> Tuple[str, Dict[str, int]]:
        counts = {"removed_periods": 0, "fixed_capitalization": 0}
        orig = text or ""
        t = orig.strip()
        # remove single end period
        new_t, n = self._end_period.subn("", t)
        if n > 0:
            counts["removed_periods"] += n
            t = new_t
        # simple capitalization: lowercase then uppercase first letter if sentence-like
        if t:
            fixed = t[0].upper() + t[1:]
            if fixed != t:
                counts["fixed_capitalization"] += 1
                t = fixed
        return t, counts

    def _clean_equation(self, eq: str) -> Tuple[str, Dict[str, int]]:
        counts = {"normalized_whitespace": 0}
        orig = eq or ""
        t = self._multi_space.sub(" ", orig).strip()
        if t != orig:
            counts["normalized_whitespace"] = 1
        return t, counts

    def clean_record(self, record: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, int]]:
        """Clean a single record (periods, whitespace, capitals)."""
        counts: Dict[str, int] = {"removed_periods": 0, "normalized_whitespace": 0, "fixed_capitalization": 0}
        new_record = dict(record)
        # Keep all original keys. Only modify fields we know.
        spoken = record.get("spoken_English")
        if spoken is not None:
            cleaned, c = self._clean_spoken(spoken)
            new_record["spoken_English"] = cleaned
            for k, v in c.items():
                counts[k] += v
        eq = record.get("equation")
        if eq is not None:
            cleaned_eq, c2 = self._clean_equation(eq)
            new_record["equation"] = cleaned_eq
            for k, v in c2.items():
                counts[k] += v
        return new_record, counts

    def clean_batch(self, records: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
        """Clean a batch of records."""
        total = {"removed_periods": 0, "normalized_whitespace": 0, "fixed_capitalization": 0}
        cleaned: List[Dict[str, str]] = []
        for r in records:
            nr, c = self.clean_record(r)
            cleaned.append(nr)
            for k in total:
                total[k] += c.get(k, 0)
        return cleaned, total
