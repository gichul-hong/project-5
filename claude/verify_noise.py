"""ref/dataset/test_noise_only 데이터가 어떤 노이즈 연산으로 만들어졌는지 역검증한다.

test_label(clean)과 test_noise_only(noisy), noise_meta.json(선언된 type/sigma)가
모두 주어져 있으므로, 노트북에 구현할 NoiseSimulator 의 수식이 실제 생성기와
동일한지 통계적으로 확인할 수 있다.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
LABEL_DIR = str(PROJECT / "ref" / "dataset" / "test_label")
NOISY_DIR = str(PROJECT / "ref" / "dataset" / "test_noise_only" / "test_noise_only")
META = os.path.join(NOISY_DIR, "noise_meta.json")


def main() -> None:
    with open(META, encoding="utf-8") as f:
        meta = json.load(f)

    labels = {os.path.basename(p): p for p in glob.glob(os.path.join(LABEL_DIR, "*.npy"))}
    stats: dict[str, list[tuple[float, float]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)

    for rec in meta:
        name, ntype, sigma = rec["file"], rec["noise_type"], float(rec["sigma"])
        counts[ntype] += 1
        lab = np.load(labels[name]).astype(np.float64)
        noi = np.load(os.path.join(NOISY_DIR, name)).astype(np.float64)
        res = noi - lab

        if ntype == "gaussian":
            # noisy = label + N(0, sigma)  ->  std(res) == sigma
            est = res.std()
        elif ntype == "rician":
            # noisy = |label + sr + i*si|  ->  E[noisy^2] = label^2 + 2 sigma^2
            est = np.sqrt(max((noi**2 - lab**2).mean(), 0.0) / 2.0)
        elif ntype == "uniform":
            # res ~ U(-sigma, sigma)  ->  std(res) == sigma / sqrt(3)
            est = res.std() * np.sqrt(3.0)
        elif ntype == "salt_and_pepper":
            # sigma/2 는 salt(=label.max()), sigma/2 는 pepper(=0) 로 덮인 픽셀 비율
            hit = np.count_nonzero(res != 0.0) / res.size
            est = hit
        else:
            raise ValueError(ntype)

        stats[ntype].append((sigma, float(est)))

    print(f"{'noise_type':<18}{'n':>4}{'declared σ (mean)':>20}{'estimated':>12}{'max |err|':>12}")
    print("-" * 66)
    for ntype in sorted(stats):
        pairs = stats[ntype]
        dec = np.array([p[0] for p in pairs])
        est = np.array([p[1] for p in pairs])
        err = np.abs(dec - est)
        # salt&pepper 는 덮인 픽셀 비율이 sigma 보다 약간 작다(겹침/원본과 동일값)
        print(f"{ntype:<18}{len(pairs):>4}{dec.mean():>20.5f}{est.mean():>12.5f}{err.max():>12.5f}")

    print("\n샘플별 상세 (앞 3개씩):")
    for ntype in sorted(stats):
        for dec, est in stats[ntype][:3]:
            print(f"  {ntype:<18} declared={dec:.5f}  estimated={est:.5f}  ratio={est / dec:.4f}")


if __name__ == "__main__":
    main()
