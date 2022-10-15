from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Domain:
    name: str
    id: int
    default_num_classes: Optional[int] = None


@dataclass(frozen=True)
class MultiDomain:
    TRAIN: str = "train"
    VAL: str = "val"
    TEST: str = "test"

    # train dataframe field
    FILEPATH: str = "filepath"
    DOMAIN_ID: str = "domain_id"
    LABEL_ID: str = "label_id"
    TARGET_FIELDS: Tuple[str, ...] = ("filepath", "domain_id", "label_id")

    # metric setting
    KNN_SAMPLES: int = 5
    METRIC_MODE: str = "mP"  # mean precision

    # each domain_id
    PRODUCT_10K: Domain = Domain(name="products_10k", id=0)
    GLR: Domain = Domain(name="glr", id=1)
    OMNI_BENCH: Domain = Domain(name="omini_bench_food", id=2, default_num_classes=673)
    MET: Domain = Domain(name="met", id=3)
    IFOOD: Domain = Domain(name="ifood", id=4)
    HOTEL_ID: Domain = Domain(name="hotel_id", id=5)
    IN_SHOP: Domain = Domain(name="in_shop", id=6)
    SOP: Domain = Domain(name="sop", id=7)
    OTHER: Domain = Domain(name="other", id=8)
    domain_name_to_id: Dict[str, int] = field(default_factory=dict)
    all_domains: List[Domain] = field(default_factory=list)

    def __post_init__(self) -> None:
        name_checks = []
        id_checks = []
        for attr_name, value in self.__dict__.items():
            if isinstance(value, Domain):
                self.all_domains.append(value)
                self.domain_name_to_id[value.name] = value.id

                name_checks.append(value.name)
                id_checks.append(value.id)

        # unique check
        assert len(name_checks) == len(set(name_checks)), name_checks
        assert len(id_checks) == len(set(id_checks)), id_checks
        # is consistent
        assert list(range(len(id_checks))) == list(self.domain_name_to_id.values()), id_checks


md_const = MultiDomain()
