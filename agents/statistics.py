"""Deterministic descriptive and categorical inference; never LLM scores."""
import math
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Geçerli pay/payda gerekli.")
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return max(0, center-half), min(1, center+half)


def descriptive_statistics(frame, metadata):
    rows = []
    for meta in metadata:
        if meta["type"] != "number" or meta["role"] != "metric":
            continue
        values = pd.to_numeric(frame[meta["name"]], errors="coerce").replace([np.inf, -np.inf], np.nan)
        rows.append({"Sütun": meta["name"], "Birim": meta["unit"], "Geçerli": int(values.count()),
                     "Eksik": int(values.isna().sum()), "Ortalama": values.mean(),
                     "Medyan": values.median(), "Standart sapma": values.std(ddof=1),
                     "Minimum": values.min(), "Maksimum": values.max()})
    return pd.DataFrame(rows)


def missing_statistics(frame):
    return pd.DataFrame({"Sütun": frame.columns, "Eksik": frame.isna().sum().values,
                         "Eksik (%)": frame.isna().mean().mul(100).values})


def categorical_test(table, independent=False):
    if not independent:
        raise ValueError("Bağımsız gözlem birimi doğrulanmadan istatistiksel test yapılmaz.")
    values = np.asarray(table, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("En az 2x2, sonlu ve negatif olmayan tam sayılı frekans tablosu gerekli.")
    values = values[values.sum(axis=1) > 0][:, values.sum(axis=0) > 0]
    if min(values.shape) < 2:
        raise ValueError("Karşılaştırılacak en az iki grup ve iki sonuç gerekli.")
    chi2, p, dof, expected = chi2_contingency(values, correction=False)
    n = int(values.sum())
    phi2 = chi2/n
    r, c = values.shape
    corrected = max(0, phi2 - (c-1)*(r-1)/(n-1)) if n>1 else 0
    divisor = min(c-1-(c-1)**2/(n-1), r-1-(r-1)**2/(n-1)) if n>1 else 0
    effect = math.sqrt(corrected/divisor) if divisor>0 else None
    method = "Ki-kare bağımsızlık testi"
    if (expected < 5).any():
        if values.shape == (2, 2):
            _, p = fisher_exact(values)
            method = "Fisher exact testi"
        else:
            return {"status": "insufficient", "reason": "Beklenen hücre sayıları 5'in altında; ki-kare yaklaşımı kullanılmadı.", "n": n}
    result = {"status": "success", "method": method, "p_value": float(p), "n": n,
              "cramers_v_corrected": effect, "minimum_expected": float(expected.min()),
              "interpretation": "İlişki için kanıt var" if p < .05 else "H0 reddedilemedi",
              "warning": "p-değeri hipotezin doğru olma olasılığı değildir. İlişki nedensellik veya temsilî örneklem kanıtı değildir."}
    if values.shape == (2, 2):
        a, b = values[:, 0]
        n1, n2 = values.sum(axis=1)
        p1, p2 = a/n1, b/n2
        l1, u1 = wilson_interval(a, n1)
        l2, u2 = wilson_interval(b, n2)
        diff = p1-p2
        result.update({"proportion_difference": float(diff),
                       "difference_ci95": [float(diff-math.sqrt((p1-l1)**2+(u2-p2)**2)),
                                           float(diff+math.sqrt((u1-p1)**2+(p2-l2)**2))],
                       "interval_method": "Newcombe (Wilson), bağımsız iki oran"})
    return result
