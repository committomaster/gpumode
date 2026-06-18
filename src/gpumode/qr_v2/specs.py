"""Official qr_v2 correctness and benchmark specs."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QrV2Spec:
    batch: int
    n: int
    cond: int
    seed: int
    case: str = "dense"

    def kwargs(self) -> dict[str, int | str]:
        return {
            "batch": self.batch,
            "n": self.n,
            "cond": self.cond,
            "seed": self.seed,
            "case": self.case,
        }

    def eval_line(self, *, include_dense_case: bool = False) -> str:
        parts = [
            f"batch:{self.batch}",
            f"n:{self.n}",
            f"cond:{self.cond}",
            f"seed:{self.seed}",
        ]
        if include_dense_case or self.case != "dense":
            parts.append(f"case:{self.case}")
        return ";".join(parts)


TEST_SPECS: tuple[QrV2Spec, ...] = (
    QrV2Spec(batch=20, n=32, cond=1, seed=53124),
    QrV2Spec(batch=40, n=176, cond=1, seed=3321),
    QrV2Spec(batch=40, n=352, cond=1, seed=1200),
    QrV2Spec(batch=16, n=512, cond=2, seed=32523),
    QrV2Spec(batch=4, n=1024, cond=2, seed=4327),
    QrV2Spec(batch=1, n=4096, cond=1, seed=75342),
    QrV2Spec(batch=16, n=512, cond=4, seed=32524, case="dense"),
    QrV2Spec(batch=16, n=512, cond=0, seed=32525, case="rankdef"),
    QrV2Spec(batch=16, n=512, cond=0, seed=32526, case="clustered"),
    QrV2Spec(batch=16, n=512, cond=0, seed=32527, case="band"),
    QrV2Spec(batch=16, n=512, cond=0, seed=32528, case="rowscale"),
    QrV2Spec(batch=16, n=512, cond=0, seed=32529, case="nearcollinear"),
    QrV2Spec(batch=4, n=1024, cond=4, seed=4328, case="dense"),
    QrV2Spec(batch=4, n=1024, cond=0, seed=4329, case="rankdef"),
    QrV2Spec(batch=4, n=1024, cond=0, seed=4330, case="nearrank"),
    QrV2Spec(batch=4, n=1024, cond=0, seed=4331, case="clustered"),
    QrV2Spec(batch=2, n=2048, cond=2, seed=224466, case="dense"),
    QrV2Spec(batch=2, n=2048, cond=0, seed=224467, case="rankdef"),
    QrV2Spec(batch=1, n=4096, cond=0, seed=75343, case="upper"),
    QrV2Spec(batch=16, n=512, cond=2, seed=32530, case="mixed"),
    QrV2Spec(batch=4, n=1024, cond=2, seed=4332, case="mixed"),
    QrV2Spec(batch=2, n=2048, cond=2, seed=224468, case="mixed"),
)


BENCHMARK_SPECS: tuple[QrV2Spec, ...] = (
    QrV2Spec(batch=20, n=32, cond=1, seed=43214),
    QrV2Spec(batch=40, n=176, cond=1, seed=423011),
    QrV2Spec(batch=40, n=352, cond=1, seed=123456),
    QrV2Spec(batch=640, n=512, cond=2, seed=1029),
    QrV2Spec(batch=60, n=1024, cond=2, seed=75342),
    QrV2Spec(batch=8, n=2048, cond=1, seed=224466),
    QrV2Spec(batch=2, n=4096, cond=1, seed=32412),
    QrV2Spec(batch=640, n=512, cond=2, seed=770001, case="mixed"),
    QrV2Spec(batch=60, n=1024, cond=2, seed=770002, case="mixed"),
    QrV2Spec(batch=640, n=512, cond=0, seed=770003, case="rankdef"),
    QrV2Spec(batch=640, n=512, cond=0, seed=770004, case="clustered"),
    QrV2Spec(batch=60, n=1024, cond=0, seed=770005, case="nearrank"),
)
