"""Schema and tensorisation for participant-reported UKB questionnaires."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
FIELD_NAMES_PATH = (
    REPO_ROOT
    / "configs"
    / "ukb"
    / "field_sets"
    / "questionnaire_model_fields_v1.json"
)

QUESTIONNAIRE_FIELD_NAMES: dict[str, str] = json.loads(FIELD_NAMES_PATH.read_text())
QUESTIONNAIRE_FIELD_IDS = tuple(QUESTIONNAIRE_FIELD_NAMES)

CONTINUOUS_FIELD_IDS = frozenset(
    {
        "884",
        "904",
        "1050",
        "1060",
        "1070",
        "1080",
        "1160",
        "1309",
        "1319",
        "1488",
        "1498",
        "20022",
        "20161",
        "2714",
        "3581",
    }
)
MULTISELECT_FIELD_IDS = frozenset(
    {"6138", "6142", "6145", "6160", "6162", "6164", "20107", "20110", "20111"}
)
ORDINAL_FIELD_IDS = frozenset(
    {
        "1697",
        "1757",
        "1170",
        "1190",
        "2306",
        "1289",
        "1299",
        "1329",
        "1339",
        "1359",
        "1379",
        "1389",
        "1528",
        "1920",
        "1930",
        "1940",
        "1950",
        "1960",
        "1970",
        "1980",
        "1990",
        "2000",
        "2010",
        "2030",
        "2050",
        "2060",
        "2070",
        "2080",
        "738",
        "924",
        "1180",
        "1200",
        "1210",
        "1220",
        "1349",
        "1369",
        "1408",
        "1478",
        "1558",
        "1687",
        "2020",
        "2090",
        "2178",
        "4598",
        "20116",
        "20117",
        "20126",
    }
)
FEMALE_ONLY_FIELD_IDS = frozenset({"2714", "2724", "3581"})

FIELD_TYPE_TO_ID = {
    "nominal": 0,
    "ordinal": 1,
    "continuous": 2,
    "multiselect": 3,
}

QUESTIONNAIRE_SUBGROUP_TO_ID = {
    "lifestyle_and_behaviour": 0,
    "family_history": 1,
    "socioeconomic_and_environmental": 2,
    "symptoms_and_disease_status": 3,
    "early_life": 4,
    "female_reproductive_and_hormonal_history": 5,
}

QUESTIONNAIRE_SUBGROUP_FIELD_IDS = {
    "lifestyle_and_behaviour": frozenset(
        {
            "884",
            "904",
            "924",
            "1031",
            "1050",
            "1060",
            "1070",
            "1080",
            "1160",
            "1170",
            "1180",
            "1190",
            "1200",
            "1210",
            "1220",
            "1289",
            "1299",
            "1309",
            "1319",
            "1329",
            "1339",
            "1349",
            "1359",
            "1369",
            "1379",
            "1389",
            "1408",
            "1418",
            "1478",
            "1488",
            "1498",
            "1528",
            "1538",
            "1558",
            "6160",
            "6162",
            "6164",
            "20116",
            "20117",
            "20161",
        }
    ),
    "family_history": frozenset({"20107", "20110", "20111"}),
    "socioeconomic_and_environmental": frozenset({"738", "6138", "6142"}),
    "symptoms_and_disease_status": frozenset(
        {
            "1757",
            "1920",
            "1930",
            "1940",
            "1950",
            "1960",
            "1970",
            "1980",
            "1990",
            "2000",
            "2010",
            "2020",
            "2030",
            "2050",
            "2060",
            "2070",
            "2080",
            "2090",
            "2100",
            "2178",
            "2188",
            "2296",
            "2306",
            "4598",
            "6145",
            "20126",
        }
    ),
    "early_life": frozenset(
        {"120", "1677", "1687", "1697", "1767", "1777", "1787", "20022"}
    ),
    "female_reproductive_and_hormonal_history": frozenset(
        {"2714", "2724", "3581"}
    ),
}

QUESTIONNAIRE_SUBGROUP_TO_ID_SEVEN = {
    "lifestyle_and_behaviour": 0,
    "family_history": 1,
    "socioeconomic_and_environmental": 2,
    "mental_health_and_psychosocial": 3,
    "general_health_and_functional_status": 4,
    "early_life": 5,
    "female_reproductive_and_hormonal_history": 6,
}

QUESTIONNAIRE_SUBGROUP_FIELD_IDS_SEVEN = {
    "lifestyle_and_behaviour": QUESTIONNAIRE_SUBGROUP_FIELD_IDS[
        "lifestyle_and_behaviour"
    ],
    "family_history": QUESTIONNAIRE_SUBGROUP_FIELD_IDS["family_history"],
    "socioeconomic_and_environmental": QUESTIONNAIRE_SUBGROUP_FIELD_IDS[
        "socioeconomic_and_environmental"
    ],
    "mental_health_and_psychosocial": frozenset(
        {
            "1920",
            "1930",
            "1940",
            "1950",
            "1960",
            "1970",
            "1980",
            "1990",
            "2000",
            "2010",
            "2020",
            "2030",
            "2050",
            "2060",
            "2070",
            "2090",
            "2100",
            "4598",
            "6145",
            "20126",
        }
    ),
    "general_health_and_functional_status": frozenset(
        {"1757", "2080", "2178", "2188", "2296", "2306"}
    ),
    "early_life": QUESTIONNAIRE_SUBGROUP_FIELD_IDS["early_life"],
    "female_reproductive_and_hormonal_history": QUESTIONNAIRE_SUBGROUP_FIELD_IDS[
        "female_reproductive_and_hormonal_history"
    ],
}

QUESTIONNAIRE_SUBGROUP_SCHEMES = {
    "six": (QUESTIONNAIRE_SUBGROUP_TO_ID, QUESTIONNAIRE_SUBGROUP_FIELD_IDS),
    "seven": (
        QUESTIONNAIRE_SUBGROUP_TO_ID_SEVEN,
        QUESTIONNAIRE_SUBGROUP_FIELD_IDS_SEVEN,
    ),
}

MISSING_STATE_TO_ID = {
    "observed": 0,
    "do_not_know": 1,
    "prefer_not_to_answer": 2,
    "question_not_presented": 3,
    "structurally_inapplicable": 4,
    "missing_assessment": 5,
    "missing_response": 6,
    "field_dropout": 7,
}
NUM_MISSING_STATES = len(MISSING_STATE_TO_ID)

_FIELD_COLUMN_RE = re.compile(r"^f\.(?P<field_id>\d+)\.(?P<instance>\d+)\.(?P<array>\d+)$")


def questionnaire_field_type(field_id: str) -> str:
    if field_id in CONTINUOUS_FIELD_IDS:
        return "continuous"
    if field_id in MULTISELECT_FIELD_IDS:
        return "multiselect"
    if field_id in ORDINAL_FIELD_IDS:
        return "ordinal"
    return "nominal"


def questionnaire_type_ids() -> list[int]:
    return [FIELD_TYPE_TO_ID[questionnaire_field_type(fid)] for fid in QUESTIONNAIRE_FIELD_IDS]


def questionnaire_subgroup_ids(scheme: str = "six") -> list[int]:
    if scheme not in QUESTIONNAIRE_SUBGROUP_SCHEMES:
        raise ValueError(
            f"Unknown questionnaire subgroup scheme {scheme!r}; "
            f"expected one of {sorted(QUESTIONNAIRE_SUBGROUP_SCHEMES)}"
        )
    subgroup_to_id, subgroup_field_ids = QUESTIONNAIRE_SUBGROUP_SCHEMES[scheme]
    field_to_subgroup = {}
    for subgroup, field_ids in subgroup_field_ids.items():
        for field_id in field_ids:
            if field_id in field_to_subgroup:
                raise ValueError(
                    f"Questionnaire field {field_id} belongs to multiple subgroups"
                )
            field_to_subgroup[field_id] = subgroup_to_id[subgroup]

    expected = set(QUESTIONNAIRE_FIELD_IDS)
    assigned = set(field_to_subgroup)
    if assigned != expected:
        missing = sorted(expected - assigned)
        extra = sorted(assigned - expected)
        raise ValueError(
            "Questionnaire subgroup mapping does not match the field schema: "
            f"missing={missing}, extra={extra}"
        )
    return [field_to_subgroup[field_id] for field_id in QUESTIONNAIRE_FIELD_IDS]


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _as_float(value) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_questionnaire_records(
    patient_data,
    assessment_dates: Sequence[object],
    patient_gender,
) -> list[dict[str, dict[str, object]]]:
    """Build leakage-safe questionnaire records for assessment instances 0-3."""
    columns_by_field_instance: dict[tuple[str, int], list[tuple[int, str]]] = {}
    requested_ids = set(QUESTIONNAIRE_FIELD_IDS)
    for column in patient_data.index:
        match = _FIELD_COLUMN_RE.match(str(column))
        if not match or match.group("field_id") not in requested_ids:
            continue
        key = (match.group("field_id"), int(match.group("instance")))
        columns_by_field_instance.setdefault(key, []).append(
            (int(match.group("array")), str(column))
        )

    is_male = _as_float(patient_gender) == 1.0
    records = []
    for target_instance in range(4):
        instance_record: dict[str, dict[str, object]] = {}
        for field_id in QUESTIONNAIRE_FIELD_IDS:
            if field_id in FEMALE_ONLY_FIELD_IDS and is_male:
                instance_record[field_id] = {
                    "values": [],
                    "state": MISSING_STATE_TO_ID["structurally_inapplicable"],
                    "source_instance": None,
                }
                continue
            if target_instance >= len(assessment_dates) or assessment_dates[target_instance] is None:
                instance_record[field_id] = {
                    "values": [],
                    "state": MISSING_STATE_TO_ID["missing_assessment"],
                    "source_instance": None,
                }
                continue

            had_presented_column = False
            selected_values: list[float] | None = None
            selected_instance = None
            for source_instance in range(target_instance, -1, -1):
                columns = sorted(
                    columns_by_field_instance.get((field_id, source_instance), []),
                    key=lambda item: item[0],
                )
                if not columns:
                    continue
                had_presented_column = True
                values = [
                    value
                    for _, column in columns
                    if (value := _as_float(patient_data.get(column))) is not None
                ]
                if values:
                    selected_values = values
                    selected_instance = source_instance
                    break

            if selected_values is None:
                state_name = "missing_response" if had_presented_column else "question_not_presented"
                instance_record[field_id] = {
                    "values": [],
                    "state": MISSING_STATE_TO_ID[state_name],
                    "source_instance": None,
                }
                continue

            if len(selected_values) == 1 and selected_values[0] in {-1.0, -3.0}:
                state_name = (
                    "do_not_know" if selected_values[0] == -1.0 else "prefer_not_to_answer"
                )
                instance_record[field_id] = {
                    "values": [],
                    "state": MISSING_STATE_TO_ID[state_name],
                    "source_instance": selected_instance,
                }
                continue

            instance_record[field_id] = {
                "values": selected_values,
                "state": MISSING_STATE_TO_ID["observed"],
                "source_instance": selected_instance,
            }
        records.append(instance_record)
    return records


def parse_questionnaire_record(value) -> dict[str, dict[str, object]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


@dataclass
class QuestionnaireTensorizer:
    """Fit train-only category vocabularies/statistics and produce fixed tensors."""

    category_maps: dict[str, dict[float, int]]
    value_stats: dict[str, tuple[float, float]]
    max_choices: int

    @classmethod
    def fit(cls, records: Sequence[object], max_choices: int = 16):
        observed: dict[str, list[float]] = {fid: [] for fid in QUESTIONNAIRE_FIELD_IDS}
        for raw_record in records:
            record = parse_questionnaire_record(raw_record)
            for field_id in QUESTIONNAIRE_FIELD_IDS:
                entry = record.get(field_id, {})
                if entry.get("state") != MISSING_STATE_TO_ID["observed"]:
                    continue
                observed[field_id].extend(
                    value
                    for raw_value in entry.get("values", [])
                    if (value := _as_float(raw_value)) is not None
                )

        category_maps = {}
        # ID 0 is padding and ID 1 is a shared out-of-vocabulary response.
        next_category_id = 2
        value_stats = {}
        for field_id in QUESTIONNAIRE_FIELD_IDS:
            values = observed[field_id]
            field_type = questionnaire_field_type(field_id)
            if field_type != "continuous":
                mapping = {}
                for value in sorted(set(values)):
                    mapping[value] = next_category_id
                    next_category_id += 1
                category_maps[field_id] = mapping
            if field_type in {"continuous", "ordinal"}:
                finite = np.asarray(values, dtype=np.float64)
                finite = finite[np.isfinite(finite)]
                mean = float(finite.mean()) if len(finite) else 0.0
                std = float(finite.std()) if len(finite) else 1.0
                value_stats[field_id] = (mean, std if std > 1e-8 else 1.0)

        return cls(
            category_maps=category_maps,
            value_stats=value_stats,
            max_choices=max_choices,
        )

    @property
    def category_vocab_size(self) -> int:
        largest = max(
            (category_id for mapping in self.category_maps.values() for category_id in mapping.values()),
            default=1,
        )
        return largest + 1

    def transform(self, raw_record) -> torch.Tensor:
        record = parse_questionnaire_record(raw_record)
        tensor = torch.zeros(
            len(QUESTIONNAIRE_FIELD_IDS),
            self.max_choices,
            3,
            dtype=torch.float32,
        )
        missing_response = MISSING_STATE_TO_ID["missing_response"]
        for field_index, field_id in enumerate(QUESTIONNAIRE_FIELD_IDS):
            entry = record.get(field_id, {})
            state = int(entry.get("state", missing_response))
            tensor[field_index, 0, 2] = state
            if state != MISSING_STATE_TO_ID["observed"]:
                continue

            values = [
                value
                for raw_value in entry.get("values", [])[: self.max_choices]
                if (value := _as_float(raw_value)) is not None
            ]
            if not values:
                tensor[field_index, 0, 2] = missing_response
                continue

            field_type = questionnaire_field_type(field_id)
            if field_type in {"continuous", "ordinal"}:
                mean, std = self.value_stats.get(field_id, (0.0, 1.0))
                tensor[field_index, 0, 0] = (values[0] - mean) / std
            if field_type != "continuous":
                mapping = self.category_maps.get(field_id, {})
                for choice_index, value in enumerate(values):
                    tensor[field_index, choice_index, 1] = mapping.get(value, 1)
        return tensor
