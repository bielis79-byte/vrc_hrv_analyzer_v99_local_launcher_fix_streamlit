
import re
import zipfile
import tempfile
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import pdist, squareform

try:
    import networkx as nx
except Exception:
    nx = None


st.set_page_config(page_title="VRC / HRV RRi Analyzer Pro v9.9", layout="wide")

# Fases ampliadas:
# - Basal + Basal2-Basal5 permiten seleccionar varias ventanas basales.
# - R1-R6 permiten seleccionar más de dos ventanas de recuperación.
PHASES = ["Basal"] + [f"Basal{i}" for i in range(2, 6)] + [f"E{i}" for i in range(1, 7)] + [f"R{i}" for i in range(1, 7)]
PHASE_GROUP = {
    "Basal": "Basal",
    **{f"Basal{i}": "Basal" for i in range(2, 6)},
    **{f"E{i}": "Ejercicio" for i in range(1, 7)},
    **{f"R{i}": "Recuperación" for i in range(1, 7)},
}
PHASE_COLORS = {
    "Basal": "rgba(0,150,255,0.24)",
    "Ejercicio": "rgba(255,140,0,0.20)",
    "Recuperación": "rgba(0,200,100,0.20)",
}
PHASE_LINE_COLORS = {
    "Basal": "#0096ff",
    "Ejercicio": "#ff8c00",
    "Recuperación": "#00c864",
}

FS_INTERP = 4.0
LAMBDA_DEFAULT = 500

PARAM_GROUPS = {
    "Tiempo": ["MeanHR", "MeanRR", "SDNN", "RMSSD", "pNN50", "SD1", "SD2"],
    "Frecuencia": ["VLF", "LF", "HF", "TOTAL", "LF_HF"],
    "Complejidad": ["DFA_alpha1", "DFA_alpha2", "ApEn", "SampEn"],
    "MSE 1-20": [f"MSE{i}" for i in range(1, 21)],
    "Recurrencia": ["REC", "DET", "Lmean", "Lmax", "ShanEn"],
}
DEFAULT_MULTI = ["RMSSD", "SDNN", "SD1", "SD2", "LF", "HF"]

DOMAIN_GROUPS = {
    "Amplitud": ["SDNN", "SD2", "TOTAL"],
    "Vagal": ["RMSSD", "SD1", "HF", "pNN50"],
    "Complejidad": ["DFA_alpha1", "DFA_alpha2", "ApEn", "SampEn"],
    "MSE 1-20": [f"MSE{i}" for i in range(1, 21)],
    "Recurrencia": ["REC", "DET", "Lmean", "Lmax", "ShanEn"],
}

MSE_COLUMNS = [f"MSE{i}" for i in range(1, 21)]


def sanitize_name(name):
    name = Path(str(name)).stem
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")
    return name or "registro"



def extract_datetime_from_name(name):
    """
    Extrae fecha/hora desde nombres de archivo.

    Admite:
    - papa_2026-06-15_17-25-11
    - papa_2026-06-15_17-01-40
    - papa_026-04-14_17-33-22  -> interpreta 026 como 2026
    - 2026-06-12 11-24-39
    - 20220615...
    """
    txt = str(name)

    patterns = [
        # yyyy-mm-dd_hh-mm-ss
        r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})[ _-](\d{1,2})[-_](\d{1,2})[-_](\d{1,2})",
        # yyy-mm-dd_hh-mm-ss cuando por truncado aparece 026-...
        r"(?<!\d)(\d{3})[-_](\d{1,2})[-_](\d{1,2})[ _-](\d{1,2})[-_](\d{1,2})[-_](\d{1,2})",
        # yyyy-mm-dd
        r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})",
        # yyy-mm-dd
        r"(?<!\d)(\d{3})[-_](\d{1,2})[-_](\d{1,2})",
        # yyyymmdd_hhmmss o yyyymmdd
        r"(20\d{2})(\d{2})(\d{2})[ _-]?(\d{2})?(\d{2})?(\d{2})?",
    ]

    for pat in patterns:
        m = re.search(pat, txt)
        if not m:
            continue

        groups = [g for g in m.groups()]
        try:
            y = int(groups[0])
            if 0 <= y < 1000:
                # ejemplo 026 -> 2026
                y = 2000 + y

            mo = int(groups[1])
            d = int(groups[2])

            h = int(groups[3]) if len(groups) > 3 and groups[3] not in [None, ""] else 0
            mi = int(groups[4]) if len(groups) > 4 and groups[4] not in [None, ""] else 0
            s = int(groups[5]) if len(groups) > 5 and groups[5] not in [None, ""] else 0

            if 2000 <= y <= 2099 and 1 <= mo <= 12 and 1 <= d <= 31:
                return pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi, second=s)
        except Exception:
            pass

    return pd.Timestamp.max


def sort_records_chronologically(record_data):
    return dict(sorted(
        record_data.items(),
        key=lambda kv: (extract_datetime_from_name(kv[0]), kv[0])
    ))


def read_rri_file(uploaded_file):
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    vals = []
    for line in text.replace(";", "\n").replace("\t", "\n").splitlines():
        line = line.strip().replace(",", ".")
        if not line:
            continue
        for p in line.split():
            try:
                vals.append(float(p))
            except Exception:
                pass

    rr = np.asarray(vals, dtype=float)
    rr = rr[np.isfinite(rr)]

    if len(rr) == 0:
        raise ValueError("No se han detectado RRi numéricos.")

    if np.nanmedian(rr) > 10:
        rr = rr / 1000.0

    rr = rr[(rr >= 0.3) & (rr <= 2.0)]

    if len(rr) == 0:
        raise ValueError("Tras el filtrado fisiológico no quedan RRi válidos.")

    return rr


def correct_artifacts_kubios_like(rr, level="none", window=5):
    rr = np.asarray(rr, dtype=float)
    rr_corr = rr.copy()
    n = len(rr)

    if level == "none" or n < 10:
        return rr_corr, np.zeros(n, dtype=bool), {
            "level": level,
            "n_artifacts": 0,
            "percent_artifacts": 0.0,
        }

    thresholds = {
        "very low": 0.45,
        "low": 0.35,
        "medium": 0.25,
        "strong": 0.15,
        "very strong": 0.05,
    }
    th = thresholds.get(level, 0.25)

    local = pd.Series(rr).rolling(window=window, center=True, min_periods=1).median().to_numpy()
    artifacts = np.abs(rr - local) > th

    if np.mean(artifacts) > 0.30:
        artifacts[:] = False

    idx = np.arange(n)
    good = ~artifacts

    if np.sum(good) >= 2 and np.sum(artifacts) > 0:
        rr_corr[artifacts] = np.interp(idx[artifacts], idx[good], rr[good])

    return rr_corr, artifacts, {
        "level": level,
        "n_artifacts": int(np.sum(artifacts)),
        "percent_artifacts": float(100 * np.mean(artifacts)),
    }


def cumulative_time(rr):
    return np.cumsum(rr)


def sec_to_hms(seconds):
    seconds = int(round(float(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def hms_to_sec(s):
    parts = [float(p) for p in str(s).strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def cut_segment(rr, start_s, end_s):
    t = cumulative_time(rr)
    return rr[(t >= start_s) & (t <= end_s)]


def empty_windows():
    return {ph: None for ph in PHASES}


def default_windows(t_max):
    """
    Autodivisión flexible del registro.

    Compatible con fases ampliadas:
    Basal, Basal2-Basal5, E1-E6, R1-R6.

    Por defecto:
    - Basal ocupa los primeros 5 min si el registro lo permite.
    - E1-E6 cubren el bloque intermedio.
    - R1-R6 cubren la parte final.
    - Basal2-Basal5 quedan vacías para que el usuario pueda definirlas manualmente.
    """
    t_max = float(max(t_max, 1.0))
    w = empty_windows()

    if t_max < 120:
        step = max(t_max / max(len(PHASES), 1), 10)
        for i, ph in enumerate(PHASES):
            w[ph] = [min(i * step, t_max), min((i + 1) * step, t_max)]
        return w

    # Basal principal
    basal_end = min(300.0, t_max)
    w["Basal"] = [0.0, basal_end]

    # Mantener basales adicionales vacías para edición manual
    for ph in [p for p in PHASES if p.startswith("Basal") and p != "Basal"]:
        w[ph] = None

    # Distribución del resto entre ejercicio y recuperación
    remaining_start = basal_end
    remaining = max(0.0, t_max - remaining_start)

    if remaining <= 0:
        return w

    # 60% del tiempo restante para ejercicio, 40% para recuperación
    exercise_total = remaining * 0.60
    recovery_total = remaining * 0.40

    e_step = exercise_total / 6.0 if exercise_total > 0 else 0
    for i in range(1, 7):
        w[f"E{i}"] = [
            min(remaining_start + (i - 1) * e_step, t_max),
            min(remaining_start + i * e_step, t_max),
        ]

    r_start = remaining_start + exercise_total
    r_step = recovery_total / 6.0 if recovery_total > 0 else 0
    for i in range(1, 7):
        w[f"R{i}"] = [
            min(r_start + (i - 1) * r_step, t_max),
            min(r_start + i * r_step, t_max),
        ]

    return w


def smoothness_priors_detrend(y, lam=500):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y - np.mean(y) if n else y

    I = sparse.eye(n, format="csc")
    e = np.ones(n)
    D2 = sparse.diags([e[:-2], -2 * e[:-2], e[:-2]], [0, 1, 2], shape=(n - 2, n), format="csc")
    trend = spsolve(I + (lam ** 2) * (D2.T @ D2), y)
    return y - trend


def interpolate_rr(rr, fs=FS_INTERP, apply_lambda=False, lam=500):
    t = cumulative_time(rr)
    if len(t) < 5:
        return np.array([]), np.array([])

    t = t - t[0]
    x = rr.copy()
    keep = np.r_[True, np.diff(t) > 0]
    t, x = t[keep], x[keep]

    if len(t) < 5:
        return np.array([]), np.array([])

    ti = np.arange(0, t[-1], 1 / fs)

    if len(ti) < 5:
        return np.array([]), np.array([])

    xi = CubicSpline(t, x, bc_type="natural")(ti)

    if apply_lambda:
        xi = smoothness_priors_detrend(xi, lam)

    return ti, xi


def time_metrics(rr):
    rr_ms = rr * 1000.0
    diff = np.diff(rr_ms)
    mean_rr = np.mean(rr_ms)
    sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan
    rmssd = np.sqrt(np.mean(diff ** 2)) if len(diff) else np.nan
    nn50 = int(np.sum(np.abs(diff) > 50)) if len(diff) else 0
    pnn50 = 100 * nn50 / len(diff) if len(diff) else np.nan
    sd1 = np.sqrt(0.5) * np.std(diff, ddof=1) if len(diff) > 1 else np.nan
    sd2 = np.sqrt(max(0, 2 * sdnn ** 2 - sd1 ** 2)) if np.isfinite(sdnn) and np.isfinite(sd1) else np.nan

    return {
        "N_RRi": len(rr),
        "Duration_s": float(np.sum(rr)),
        "MeanRR": mean_rr,
        "MeanHR": 60000 / mean_rr if mean_rr > 0 else np.nan,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "NN50": nn50,
        "pNN50": pnn50,
        "SD1": sd1,
        "SD2": sd2,
    }


def psd_metrics(rr):
    ti, xi = interpolate_rr(rr, fs=FS_INTERP, apply_lambda=True, lam=LAMBDA_DEFAULT)

    if len(xi) < 32:
        return {"VLF": np.nan, "LF": np.nan, "HF": np.nan, "TOTAL": np.nan, "LF_HF": np.nan}

    xi_ms = xi * 1000
    xi_ms = xi_ms - np.mean(xi_ms)
    nperseg = min(int(256 * FS_INTERP), len(xi_ms))
    noverlap = int(0.5 * nperseg)

    f, pxx = signal.welch(
        xi_ms,
        fs=FS_INTERP,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )

    def bp(lo, hi):
        mask = (f >= lo) & (f < hi)
        return np.trapezoid(pxx[mask], f[mask]) if np.any(mask) else np.nan

    vlf, lf, hf = bp(0.0033, 0.04), bp(0.04, 0.15), bp(0.15, 0.40)
    total = np.nansum([vlf, lf, hf])

    return {"VLF": vlf, "LF": lf, "HF": hf, "TOTAL": total, "LF_HF": lf / hf if pd.notna(hf) and hf > 0 else np.nan}


def _phi_apen(x, m, r):
    n = len(x)

    if n <= m + 1:
        return np.nan

    pats = np.array([x[i:i + m] for i in range(n - m + 1)])
    vals = []

    for p in pats:
        dist = np.max(np.abs(pats - p), axis=1)
        c = np.mean(dist <= r)
        if c > 0:
            vals.append(np.log(c))

    return np.mean(vals) if vals else np.nan


def apen_calc(x, m=2, r_ratio=0.2):
    x = smoothness_priors_detrend(np.asarray(x, dtype=float), LAMBDA_DEFAULT)
    r = r_ratio * np.std(x, ddof=1)

    if not np.isfinite(r) or r == 0:
        return np.nan

    return _phi_apen(x, m, r) - _phi_apen(x, m + 1, r)


def sampen_calc(x, m=2, r_ratio=0.2):
    x = smoothness_priors_detrend(np.asarray(x, dtype=float), LAMBDA_DEFAULT)
    n = len(x)

    if n <= m + 2:
        return np.nan

    r = r_ratio * np.std(x, ddof=1)

    if not np.isfinite(r) or r == 0:
        return np.nan

    def count(mm):
        pats = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        c = 0
        for i in range(len(pats) - 1):
            dist = np.max(np.abs(pats[i + 1:] - pats[i]), axis=1)
            c += np.sum(dist <= r)
        return c

    b, a = count(m), count(m + 1)

    if a == 0 or b == 0:
        return np.nan

    return -np.log(a / b)


def dfa_calc(x):
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n < 50:
        return np.nan, np.nan

    y = np.cumsum(x - np.mean(x))
    scales = np.unique(np.floor(np.logspace(np.log10(4), np.log10(max(5, n // 4)), 18)).astype(int))

    ss, ff = [], []

    for s in scales:
        if s < 4 or n // s < 2:
            continue

        rms = []

        for i in range(n // s):
            seg = y[i * s:(i + 1) * s]
            t = np.arange(s)
            co = np.polyfit(t, seg, 1)
            rms.append(np.sqrt(np.mean((seg - np.polyval(co, t)) ** 2)))

        val = np.sqrt(np.mean(np.asarray(rms) ** 2))

        if val > 0:
            ss.append(s)
            ff.append(val)

    ss, ff = np.asarray(ss), np.asarray(ff)

    if len(ss) < 4:
        return np.nan, np.nan

    m1, m2 = (ss >= 4) & (ss <= 16), ss > 16

    return (
        np.polyfit(np.log(ss[m1]), np.log(ff[m1]), 1)[0] if np.sum(m1) >= 2 else np.nan,
        np.polyfit(np.log(ss[m2]), np.log(ff[m2]), 1)[0] if np.sum(m2) >= 2 else np.nan,
    )


def rqa_calc(x, emb_dim=10, tau=1, l_min=2, max_n=500):
    x = np.asarray(x, dtype=float)

    if len(x) > max_n:
        x = x[np.linspace(0, len(x) - 1, max_n).astype(int)]

    n = len(x) - (emb_dim - 1) * tau

    if n < 20:
        return {"REC": np.nan, "DET": np.nan, "Lmean": np.nan, "Lmax": np.nan, "ShanEn": np.nan}

    D = squareform(pdist(np.array([x[i:i + emb_dim * tau:tau] for i in range(n)])))
    radius = np.sqrt(emb_dim) * np.std(x, ddof=1)
    R = (D <= radius).astype(int)
    np.fill_diagonal(R, 0)
    rec = 100 * R.sum() / (n * n - n)

    lens = []

    for k in range(-n + 1, n):
        diag = np.diag(R, k=k)
        c = 0

        for val in diag:
            if val:
                c += 1
            else:
                if c >= l_min:
                    lens.append(c)
                c = 0

        if c >= l_min:
            lens.append(c)

    if not lens:
        return {"REC": rec, "DET": 0, "Lmean": 0, "Lmax": 0, "ShanEn": 0}

    lens = np.asarray(lens)
    det = 100 * lens.sum() / R.sum() if R.sum() > 0 else 0
    vals, counts = np.unique(lens, return_counts=True)
    p = counts / counts.sum()

    return {"REC": rec, "DET": det, "Lmean": np.mean(lens), "Lmax": np.max(lens), "ShanEn": -np.sum(p * np.log(p))}




def hvg_graph(x, max_nodes=500):
    if nx is None:
        return None

    x = np.asarray(x, dtype=float)
    if len(x) > max_nodes:
        idx = np.linspace(0, len(x) - 1, max_nodes).astype(int)
        x = x[idx]

    n = len(x)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n - 1):
        G.add_edge(i, i + 1)
        for j in range(i + 2, n):
            if np.max(x[i + 1:j]) < min(x[i], x[j]):
                G.add_edge(i, j)

    return G



def classify_hvg_graph_type(metrics):
    """
    Clasificación orientativa del tipo de grafo HVG.

    Tipos:
    - Libre de escala / jerárquico
    - Small-world funcional
    - Lineal / cadena
    - Regular / homogéneo
    - Complejo mixto
    """
    try:
        nodes = float(metrics.get("HVG_nodes", np.nan))
        edges = float(metrics.get("HVG_edges", np.nan))
        degree_mean = float(metrics.get("HVG_degree_mean", np.nan))
        degree_max = float(metrics.get("HVG_degree_max", np.nan))
        hubs = float(metrics.get("HVG_hubs_p90", np.nan))
        clustering = float(metrics.get("HVG_clustering", np.nan))
        lam = float(metrics.get("HVG_lambda", np.nan))
        path = float(metrics.get("HVG_path_length", np.nan))
        diameter = float(metrics.get("HVG_diameter", np.nan))
    except Exception:
        return {
            "HVG_graph_type": "No clasificable",
            "HVG_graph_interpretation": "No hay métricas suficientes para clasificar el grafo.",
            "HVG_graph_score_scale_free": np.nan,
            "HVG_graph_score_small_world": np.nan,
            "HVG_graph_score_chain": np.nan,
            "HVG_topology_state": "No clasificable",
            "HVG_compactness_index": np.nan,
            "HVG_topology_interpretation": "No hay métricas suficientes para valorar compactación/dispersión.",
        }

    if not np.isfinite(nodes) or nodes < 20:
        return {
            "HVG_graph_type": "No clasificable",
            "HVG_graph_interpretation": "Ventana demasiado corta o grafo insuficiente.",
            "HVG_graph_score_scale_free": np.nan,
            "HVG_graph_score_small_world": np.nan,
            "HVG_graph_score_chain": np.nan,
            "HVG_topology_state": "No clasificable",
            "HVG_compactness_index": np.nan,
            "HVG_topology_interpretation": "No hay métricas suficientes para valorar compactación/dispersión.",
        }

    edge_density = edges / max(nodes, 1)
    hub_ratio = hubs / max(nodes, 1)
    degree_contrast = degree_max / max(degree_mean, 1e-9)
    diameter_rel = diameter / max(nodes, 1) if np.isfinite(diameter) else np.nan
    path_rel = path / max(nodes, 1) if np.isfinite(path) else np.nan

    scale_free_score = 0
    if np.isfinite(degree_contrast):
        scale_free_score += min(45, 12 * degree_contrast)
    if np.isfinite(hub_ratio):
        scale_free_score += min(25, 300 * hub_ratio)
    if np.isfinite(lam):
        if lam < 0.45:
            scale_free_score += 20
        elif lam < 0.75:
            scale_free_score += 12
        elif lam < 1.1:
            scale_free_score += 6
    if np.isfinite(clustering) and clustering > 0.08:
        scale_free_score += 10
    scale_free_score = float(min(100, scale_free_score))

    small_world_score = 0
    if np.isfinite(clustering):
        small_world_score += min(45, clustering * 120)
    if np.isfinite(path_rel):
        if path_rel < 0.12:
            small_world_score += 30
        elif path_rel < 0.20:
            small_world_score += 18
        elif path_rel < 0.30:
            small_world_score += 8
    if np.isfinite(diameter_rel):
        if diameter_rel < 0.25:
            small_world_score += 20
        elif diameter_rel < 0.40:
            small_world_score += 10
    if np.isfinite(edge_density) and edge_density > 1.3:
        small_world_score += 5
    small_world_score = float(min(100, small_world_score))

    chain_score = 0
    if np.isfinite(edge_density):
        if edge_density < 1.15:
            chain_score += 40
        elif edge_density < 1.35:
            chain_score += 25
    if np.isfinite(degree_mean):
        if degree_mean < 2.4:
            chain_score += 25
        elif degree_mean < 3.0:
            chain_score += 12
    if np.isfinite(diameter_rel):
        if diameter_rel > 0.45:
            chain_score += 25
        elif diameter_rel > 0.30:
            chain_score += 12
    if np.isfinite(clustering) and clustering < 0.05:
        chain_score += 10
    chain_score = float(min(100, chain_score))

    if chain_score >= 65:
        graph_type = "Lineal / cadena"
        interp = (
            "Grafo con pocas conexiones transversales, bajo grado medio y/o diámetro relativamente alto. "
            "Sugiere una dinámica RRi más secuencial, con menor integración global."
        )
    elif scale_free_score >= 60 and scale_free_score >= small_world_score:
        graph_type = "Libre de escala / jerárquico"
        interp = (
            "Grafo con hubs relativamente marcados y distribución de grados heterogénea. "
            "Sugiere una dinámica con nodos dominantes que conectan distintas partes de la señal."
        )
    elif small_world_score >= 60:
        graph_type = "Small-world funcional"
        interp = (
            "Grafo con agrupamiento local y caminos relativamente cortos. "
            "Sugiere equilibrio entre especialización local e integración global."
        )
    elif scale_free_score >= 45 and small_world_score >= 45:
        graph_type = "Complejo mixto"
        interp = (
            "Combina rasgos de hubs y conectividad local/global. "
            "Puede indicar una organización intermedia de la dinámica RRi."
        )
    else:
        graph_type = "Regular / homogéneo"
        interp = (
            "Grafo sin hubs claramente dominantes y con conectividad relativamente homogénea. "
            "Sugiere una dinámica más uniforme o menos jerárquica."
        )

    return {
        "HVG_graph_type": graph_type,
        "HVG_graph_interpretation": interp,
        "HVG_graph_score_scale_free": round(scale_free_score, 1),
        "HVG_graph_score_small_world": round(small_world_score, 1),
        "HVG_graph_score_chain": round(chain_score, 1),
    }




# ============================================================
# INTERPRETACIÓN AVANZADA HVG / GRAFOS
# ============================================================

def _safe_float(x, default=np.nan):
    try:
        v = pd.to_numeric(x, errors="coerce")
        return float(v) if pd.notna(v) else default
    except Exception:
        return default


def hvg_reference_ranges():
    """
    Rangos orientativos para interpretación clínica/topológica.
    No son rangos diagnósticos cerrados; sirven para contextualizar.
    """
    return pd.DataFrame([
        {
            "Métrica": "HVG_clustering",
            "Qué mide": "Agrupamiento local de la red.",
            "Muy bajo": "< 0.20",
            "Bajo": "0.20 - 0.40",
            "Normal/orientativo": "0.40 - 0.70",
            "Alto": "> 0.70",
            "Lectura clínica/topológica": "Más alto = mayor compactación local y organización por vecindarios."
        },
        {
            "Métrica": "HVG_degree_mean",
            "Qué mide": "Conexiones promedio por nodo.",
            "Muy bajo": "< 2.5",
            "Bajo": "2.5 - 3.5",
            "Normal/orientativo": "3.5 - 5",
            "Alto": "> 5",
            "Lectura clínica/topológica": "Más alto = mayor conectividad global de la señal transformada en red."
        },
        {
            "Métrica": "HVG_degree_max",
            "Qué mide": "Grado del nodo más conectado.",
            "Muy bajo": "< 6",
            "Bajo": "6 - 10",
            "Normal/orientativo": "10 - 20",
            "Alto": "> 20",
            "Lectura clínica/topológica": "Valores altos indican presencia de hubs o nodos dominantes."
        },
        {
            "Métrica": "HVG_hubs_p90",
            "Qué mide": "Nodos con conectividad alta, por encima del percentil 90.",
            "Muy bajo": "< 20",
            "Bajo": "20 - 40",
            "Normal/orientativo": "40 - 80",
            "Alto": "> 80",
            "Lectura clínica/topológica": "Más hubs suelen indicar mayor centralización e integración."
        },
        {
            "Métrica": "HVG_lambda",
            "Qué mide": "Pendiente/exponente aproximado de la distribución de grados.",
            "Muy bajo": "< 0.30",
            "Bajo": "0.30 - 0.80",
            "Normal/orientativo": "0.80 - 1.50",
            "Alto": "> 1.50",
            "Lectura clínica/topológica": "Valores bajos-moderados son compatibles con cola pesada/hubs; valores altos sugieren red más homogénea."
        },
        {
            "Métrica": "HVG_path_length",
            "Qué mide": "Camino medio entre nodos.",
            "Muy bajo": "< 8",
            "Bajo": "8 - 15",
            "Normal/orientativo": "15 - 25",
            "Alto": "> 25",
            "Lectura clínica/topológica": "Menor camino medio = mejor integración global."
        },
        {
            "Métrica": "HVG_diameter",
            "Qué mide": "Distancia máxima entre dos nodos conectados.",
            "Muy bajo": "< 10",
            "Bajo": "10 - 25",
            "Normal/orientativo": "25 - 40",
            "Alto": "> 40",
            "Lectura clínica/topológica": "Diámetro menor = grafo más compacto; diámetro alto = red más dispersa."
        },
    ])


def hvg_metric_reference_label(metric, value):
    """
    Etiqueta cualitativa orientativa por métrica.
    """
    v = _safe_float(value)

    if not np.isfinite(v):
        return "No clasificable"

    if metric == "HVG_clustering":
        if v < 0.20: return "Muy bajo"
        if v < 0.40: return "Bajo"
        if v <= 0.70: return "Normal/orientativo"
        return "Alto"

    if metric == "HVG_degree_mean":
        if v < 2.5: return "Muy bajo"
        if v < 3.5: return "Bajo"
        if v <= 5: return "Normal/orientativo"
        return "Alto"

    if metric == "HVG_degree_max":
        if v < 6: return "Muy bajo"
        if v < 10: return "Bajo"
        if v <= 20: return "Normal/orientativo"
        return "Alto"

    if metric == "HVG_hubs_p90":
        if v < 20: return "Muy bajo"
        if v < 40: return "Bajo"
        if v <= 80: return "Normal/orientativo"
        return "Alto"

    if metric == "HVG_lambda":
        if v < 0.30: return "Muy bajo"
        if v < 0.80: return "Bajo/compatible hubs"
        if v <= 1.50: return "Normal/orientativo"
        return "Alto/homogéneo"

    if metric == "HVG_path_length":
        if v < 8: return "Muy bajo/compacto"
        if v < 15: return "Bajo/compacto"
        if v <= 25: return "Normal/orientativo"
        return "Alto/disperso"

    if metric == "HVG_diameter":
        if v < 10: return "Muy bajo/compacto"
        if v < 25: return "Bajo/compacto"
        if v <= 40: return "Normal/orientativo"
        return "Alto/disperso"

    return ""


def hvg_topology_state(metrics):
    """
    Clasificación compactación local vs dispersión global.

    Se combina información de:
    - clustering
    - hubs
    - grado máximo/medio
    - camino medio
    - diámetro

    Devuelve:
    - estado textual
    - índice aproximado en escala -2 a +2
    - explicación.
    """
    nodes = _safe_float(metrics.get("HVG_nodes"))
    clustering = _safe_float(metrics.get("HVG_clustering"))
    hubs = _safe_float(metrics.get("HVG_hubs_p90"))
    degree_mean = _safe_float(metrics.get("HVG_degree_mean"))
    degree_max = _safe_float(metrics.get("HVG_degree_max"))
    path = _safe_float(metrics.get("HVG_path_length"))
    diameter = _safe_float(metrics.get("HVG_diameter"))

    if not np.isfinite(nodes) or nodes <= 0:
        return {
            "HVG_topology_state": "No clasificable",
            "HVG_compactness_index": np.nan,
            "HVG_topology_interpretation": "No hay nodos suficientes para valorar compactación/dispersión."
        }

    hub_ratio = hubs / max(nodes, 1) if np.isfinite(hubs) else np.nan
    degree_contrast = degree_max / max(degree_mean, 1e-9) if np.isfinite(degree_max) and np.isfinite(degree_mean) else np.nan
    path_rel = path / max(nodes, 1) if np.isfinite(path) else np.nan
    diameter_rel = diameter / max(nodes, 1) if np.isfinite(diameter) else np.nan

    score = 0.0

    # Compactación local
    if np.isfinite(clustering):
        if clustering >= 0.70: score += 0.9
        elif clustering >= 0.50: score += 0.6
        elif clustering >= 0.35: score += 0.25
        elif clustering < 0.20: score -= 0.5

    if np.isfinite(hub_ratio):
        if hub_ratio >= 0.12: score += 0.45
        elif hub_ratio >= 0.08: score += 0.30
        elif hub_ratio < 0.04: score -= 0.25

    if np.isfinite(degree_contrast):
        if degree_contrast >= 4.0: score += 0.45
        elif degree_contrast >= 3.0: score += 0.25

    # Dispersión global
    if np.isfinite(path_rel):
        if path_rel < 0.08: score += 0.40
        elif path_rel < 0.15: score += 0.25
        elif path_rel > 0.30: score -= 0.55

    if np.isfinite(diameter_rel):
        if diameter_rel < 0.15: score += 0.45
        elif diameter_rel < 0.25: score += 0.25
        elif diameter_rel > 0.40: score -= 0.60

    score = float(np.clip(score, -2.0, 2.0))

    if score >= 1.0:
        state = "Compacto local"
        interp = (
            "Red con alta compactación local: predominan agrupamientos, hubs y distancias relativamente cortas. "
            "Sugiere una organización más integrada y centralizada."
        )
    elif score >= 0.3:
        state = "Tendencia compacta"
        interp = (
            "Red con tendencia a la compactación: conserva conectividad local/global razonable, aunque sin máxima centralización."
        )
    elif score > -0.3:
        state = "Equilibrado"
        interp = (
            "Red con equilibrio entre integración y dispersión. No predomina claramente la compactación ni la fragmentación."
        )
    elif score > -1.0:
        state = "Tendencia dispersa"
        interp = (
            "Red con tendencia a mayor dispersión: menor compactación local o caminos más largos entre nodos."
        )
    else:
        state = "Disperso global"
        interp = (
            "Red más fragmentada o menos integrada globalmente, con caminos/diámetro relativamente largos y menor centralización."
        )

    return {
        "HVG_topology_state": state,
        "HVG_compactness_index": round(score, 2),
        "HVG_topology_interpretation": interp
    }


def hvg_summary_card(metrics):
    """
    Resumen corto para mostrar encima de las tablas.
    """
    graph_type = metrics.get("HVG_graph_type", "No clasificable")
    topology = metrics.get("HVG_topology_state", "No clasificable")
    compactness = metrics.get("HVG_compactness_index", np.nan)

    scale_free = metrics.get("HVG_graph_score_scale_free", np.nan)
    small_world = metrics.get("HVG_graph_score_small_world", np.nan)
    chain = metrics.get("HVG_graph_score_chain", np.nan)

    return pd.DataFrame([
        {"Aspecto": "Tipo de grafo", "Resultado": graph_type},
        {"Aspecto": "Organización topológica", "Resultado": topology},
        {"Aspecto": "Índice compactación (-2 a +2)", "Resultado": compactness},
        {"Aspecto": "Score libre de escala (0-100)", "Resultado": scale_free},
        {"Aspecto": "Score small-world (0-100)", "Resultado": small_world},
        {"Aspecto": "Score cadena/dispersión (0-100)", "Resultado": chain},
        {"Aspecto": "Lectura compactación/dispersión", "Resultado": metrics.get("HVG_topology_interpretation", "")},
        {"Aspecto": "Lectura tipo de grafo", "Resultado": metrics.get("HVG_graph_interpretation", "")},
    ])


def hvg_reference_value_table(metrics_df):
    """
    Tabla larga con valor, rango orientativo y significado de cada métrica HVG.
    """
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()

    hvg_cols = [
        "HVG_graph_type",
        "HVG_topology_state",
        "HVG_compactness_index",
        "HVG_graph_score_scale_free",
        "HVG_graph_score_small_world",
        "HVG_graph_score_chain",
        "HVG_nodes",
        "HVG_edges",
        "HVG_degree_mean",
        "HVG_degree_max",
        "HVG_hubs_p90",
        "HVG_clustering",
        "HVG_lambda",
        "HVG_path_length",
        "HVG_diameter",
        "HVG_graph_interpretation",
        "HVG_topology_interpretation",
    ]

    explanations = {
        "HVG_graph_type": "Tipo de organización topológica dominante.",
        "HVG_topology_state": "Clasificación compactación local vs dispersión global.",
        "HVG_compactness_index": "Índice aproximado -2 a +2: valores positivos indican compactación local; negativos, dispersión global.",
        "HVG_graph_score_scale_free": "Score 0-100 de rasgos libre de escala / hubs.",
        "HVG_graph_score_small_world": "Score 0-100 de rasgos small-world: clustering + caminos cortos.",
        "HVG_graph_score_chain": "Score 0-100 de rasgos lineales/cadena.",
        "HVG_nodes": "Número de nodos analizados.",
        "HVG_edges": "Número de conexiones visibles entre nodos.",
        "HVG_degree_mean": "Conexiones promedio por nodo.",
        "HVG_degree_max": "Grado del nodo más conectado.",
        "HVG_hubs_p90": "Número de nodos con conectividad alta.",
        "HVG_clustering": "Agrupamiento local de la red.",
        "HVG_lambda": "Pendiente/exponente aproximado de la distribución de grados.",
        "HVG_path_length": "Camino medio entre nodos.",
        "HVG_diameter": "Distancia máxima entre dos nodos conectados.",
        "HVG_graph_interpretation": "Interpretación automática del tipo de grafo.",
        "HVG_topology_interpretation": "Interpretación automática de compactación/dispersión.",
    }

    rows = []
    for fase, row in metrics_df.iterrows():
        for col in hvg_cols:
            if col in metrics_df.columns:
                rows.append({
                    "Fase": fase,
                    "Métrica": col,
                    "Valor": row[col],
                    "Rango orientativo": hvg_metric_reference_label(col, row[col]),
                    "Qué significa": explanations.get(col, ""),
                })

    return pd.DataFrame(rows)


def hvg_metrics(rr, max_nodes=500):
    if nx is None:
        return {
            "HVG_nodes": np.nan,
            "HVG_edges": np.nan,
            "HVG_degree_mean": np.nan,
            "HVG_degree_max": np.nan,
            "HVG_hubs_p90": np.nan,
            "HVG_clustering": np.nan,
            "HVG_lambda": np.nan,
            "HVG_path_length": np.nan,
            "HVG_diameter": np.nan,
            "HVG_graph_type": "No clasificable",
            "HVG_graph_interpretation": "No hay métricas suficientes para clasificar el grafo.",
            "HVG_graph_score_scale_free": np.nan,
            "HVG_graph_score_small_world": np.nan,
            "HVG_graph_score_chain": np.nan,
            "HVG_topology_state": "No clasificable",
            "HVG_compactness_index": np.nan,
            "HVG_topology_interpretation": "No hay métricas suficientes para valorar compactación/dispersión.",
        }

    G = hvg_graph(rr, max_nodes=max_nodes)
    if G is None or G.number_of_nodes() < 20:
        return {
            "HVG_nodes": G.number_of_nodes() if G is not None else 0,
            "HVG_edges": np.nan,
            "HVG_degree_mean": np.nan,
            "HVG_degree_max": np.nan,
            "HVG_hubs_p90": np.nan,
            "HVG_clustering": np.nan,
            "HVG_lambda": np.nan,
            "HVG_path_length": np.nan,
            "HVG_diameter": np.nan,
            "HVG_graph_type": "No clasificable",
            "HVG_graph_interpretation": "No hay métricas suficientes para clasificar el grafo.",
            "HVG_graph_score_scale_free": np.nan,
            "HVG_graph_score_small_world": np.nan,
            "HVG_graph_score_chain": np.nan,
            "HVG_topology_state": "No clasificable",
            "HVG_compactness_index": np.nan,
            "HVG_topology_interpretation": "No hay métricas suficientes para valorar compactación/dispersión.",
        }

    n = G.number_of_nodes()
    m = G.number_of_edges()
    deg = np.array([d for _, d in G.degree()])

    vals, counts = np.unique(deg, return_counts=True)
    p = counts / counts.sum()
    mask = (vals > 1) & (p > 0)
    lam = -np.polyfit(vals[mask], np.log(p[mask]), 1)[0] if np.sum(mask) >= 2 else np.nan

    if nx.is_connected(G):
        path_length = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
    else:
        path_length = np.nan
        diameter = np.nan

    base_metrics = {
        "HVG_nodes": n,
        "HVG_edges": m,
        "HVG_degree_mean": 2 * m / n if n else np.nan,
        "HVG_degree_max": np.max(deg) if len(deg) else np.nan,
        "HVG_hubs_p90": int(np.sum(deg >= np.percentile(deg, 90))) if len(deg) else np.nan,
        "HVG_clustering": nx.average_clustering(G) if n else np.nan,
        "HVG_lambda": lam,
        "HVG_path_length": path_length,
        "HVG_diameter": diameter,
    }
    base_metrics.update(classify_hvg_graph_type(base_metrics))
    base_metrics.update(hvg_topology_state(base_metrics))
    return base_metrics


def hvg_network_figure(rr, title="HVG", max_nodes=140):
    fig = go.Figure()
    if nx is None:
        fig.update_layout(title="NetworkX no disponible")
        return fig

    G = hvg_graph(rr, max_nodes=max_nodes)
    if G is None or G.number_of_nodes() == 0:
        fig.update_layout(title="Sin grafo")
        return fig

    pos = nx.spring_layout(G, seed=42, k=0.18, iterations=60)

    edge_x, edge_y = [], []
    for a, b in G.edges():
        edge_x += [pos[a][0], pos[b][0], None]
        edge_y += [pos[a][1], pos[b][1], None]

    deg = dict(G.degree())
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_size = [6 + deg[n] * 2.5 for n in G.nodes()]
    node_text = [f"n={n}<br>grado={deg[n]}" for n in G.nodes()]

    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers", marker=dict(size=node_size), text=node_text, hoverinfo="text", showlegend=False))
    fig.update_layout(title=title, height=520, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig





def poincare_panel_figure(record_data, global_windows, record_windows, phase, use_independent):
    """
    Poincaré en paneles separados por registro, similar a grafos HVG comparativos.
    """
    records = list(record_data.keys())
    n = len(records)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(title="Sin registros")
        return fig

    cols = min(2, n)
    rows = int(np.ceil(n / cols))

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[_short_record_label(r, 30) for r in records],
        horizontal_spacing=0.08,
        vertical_spacing=0.14
    )

    global_min = np.inf
    global_max = -np.inf

    cache = {}

    for rec in records:
        windows = get_record_windows(global_windows, record_windows, rec, use_independent)
        w = windows.get(phase)
        if w is None:
            cache[rec] = None
            continue

        seg = cut_segment(record_data[rec]["rr"], w[0], w[1])
        if len(seg) < 3:
            cache[rec] = None
            continue

        rr_ms = seg * 1000
        x = rr_ms[:-1]
        y = rr_ms[1:]

        diff = np.diff(rr_ms)
        sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan
        sd1 = np.sqrt(0.5) * np.std(diff, ddof=1) if len(diff) > 1 else np.nan
        sd2 = np.sqrt(max(0, 2 * sdnn ** 2 - sd1 ** 2)) if np.isfinite(sdnn) and np.isfinite(sd1) else np.nan

        cache[rec] = (x, y, sd1, sd2)

        global_min = min(global_min, np.nanmin(x), np.nanmin(y))
        global_max = max(global_max, np.nanmax(x), np.nanmax(y))

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        fig = go.Figure()
        fig.update_layout(title=f"Poincaré {phase}: sin datos suficientes")
        return fig

    pad = max(20, 0.05 * (global_max - global_min))
    global_min -= pad
    global_max += pad

    for idx, rec in enumerate(records):
        r = idx // cols + 1
        c = idx % cols + 1
        item = cache.get(rec)

        if item is None:
            fig.add_annotation(
                text="Sin datos suficientes",
                x=0.5, y=0.5,
                xref=f"x{idx+1 if idx > 0 else ''} domain",
                yref=f"y{idx+1 if idx > 0 else ''} domain",
                showarrow=False
            )
            continue

        x, y, sd1, sd2 = item

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=5, opacity=0.62),
                name=_short_record_label(rec, 24),
                showlegend=False,
                hovertemplate="RR(n): %{x:.1f} ms<br>RR(n+1): %{y:.1f} ms<extra></extra>",
            ),
            row=r,
            col=c
        )

        # Línea identidad
        fig.add_trace(
            go.Scatter(
                x=[global_min, global_max],
                y=[global_min, global_max],
                mode="lines",
                line=dict(width=1, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=r,
            col=c
        )

        fig.add_annotation(
            text=f"SD1={sd1:.1f} ms<br>SD2={sd2:.1f} ms",
            x=0.03,
            y=0.97,
            xref=f"x{idx+1 if idx > 0 else ''} domain",
            yref=f"y{idx+1 if idx > 0 else ''} domain",
            showarrow=False,
            align="left",
            bgcolor="rgba(0,0,0,0.25)",
            bordercolor="rgba(255,255,255,0.25)",
        )

        fig.update_xaxes(range=[global_min, global_max], title_text="RR(n) ms", row=r, col=c)
        fig.update_yaxes(range=[global_min, global_max], title_text="RR(n+1) ms", row=r, col=c, scaleanchor=f"x{idx+1 if idx > 0 else ''}", scaleratio=1)

    fig.update_layout(
        height=max(560, rows * 470),
        title=f"Poincaré en paneles separados · {phase}",
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig



def hvg_network_compare_figure(record_data, global_windows, record_windows, phase, use_independent, max_nodes=120):
    """
    Muestra los grafos HVG de todos los registros en paneles comparables.
    """
    if nx is None:
        fig = go.Figure()
        fig.update_layout(title="NetworkX no disponible")
        return fig

    records = list(record_data.keys())
    n = len(records)
    if n == 0:
        return go.Figure()

    cols = min(2, n)
    rows = int(np.ceil(n / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[_short_record_label(r, 28) for r in records],
        horizontal_spacing=0.04,
        vertical_spacing=0.12
    )

    for idx, rec in enumerate(records):
        r = idx // cols + 1
        c = idx % cols + 1

        windows = get_record_windows(global_windows, record_windows, rec, use_independent)
        w = windows.get(phase)
        if w is None:
            continue

        seg = cut_segment(record_data[rec]["rr"], w[0], w[1])
        if len(seg) < 20:
            continue

        G = hvg_graph(seg, max_nodes=max_nodes)
        if G is None or G.number_of_nodes() == 0:
            continue

        pos = nx.spring_layout(G, seed=42, k=0.20, iterations=60)

        edge_x, edge_y = [], []
        for a, b in G.edges():
            edge_x += [pos[a][0], pos[b][0], None]
            edge_y += [pos[a][1], pos[b][1], None]

        deg = dict(G.degree())
        node_x = [pos[nn][0] for nn in G.nodes()]
        node_y = [pos[nn][1] for nn in G.nodes()]
        node_size = [5 + deg[nn] * 2.2 for nn in G.nodes()]
        node_text = [f"{rec}<br>n={nn}<br>grado={deg[nn]}" for nn in G.nodes()]

        fig.add_trace(
            go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=0.45),
                hoverinfo="skip",
                showlegend=False
            ),
            row=r, col=c
        )
        fig.add_trace(
            go.Scatter(
                x=node_x, y=node_y, mode="markers",
                marker=dict(size=node_size, opacity=0.82),
                text=node_text,
                hoverinfo="text",
                showlegend=False
            ),
            row=r, col=c
        )

        fig.update_xaxes(visible=False, row=r, col=c)
        fig.update_yaxes(visible=False, row=r, col=c)

    fig.update_layout(
        height=max(520, rows * 440),
        title=f"HVG comparativo · {phase}",
        margin=dict(l=20, r=20, t=70, b=20)
    )
    return fig


def poincare_figure(record_data, global_windows, record_windows, phase, use_independent):
    fig = go.Figure()

    for rec, data in record_data.items():
        windows = get_record_windows(global_windows, record_windows, rec, use_independent)
        w = windows.get(phase)
        if w is None:
            continue

        seg = cut_segment(data["rr"], w[0], w[1])
        if len(seg) < 3:
            continue

        rr_ms = seg * 1000
        x = rr_ms[:-1]
        y = rr_ms[1:]
        diff = np.diff(rr_ms)
        sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan
        sd1 = np.sqrt(0.5) * np.std(diff, ddof=1) if len(diff) > 1 else np.nan
        sd2 = np.sqrt(max(0, 2 * sdnn ** 2 - sd1 ** 2)) if np.isfinite(sdnn) and np.isfinite(sd1) else np.nan

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=f"{rec} · SD1={sd1:.1f}, SD2={sd2:.1f}",
            marker=dict(size=6, opacity=0.65)
        ))

    fig.update_layout(
        title=f"Poincaré comparativo · {phase}",
        height=560,
        xaxis_title="RR(n) ms",
        yaxis_title="RR(n+1) ms",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig






# ============================================================
# ENTROPÍAS COHERENTES: SampEn y MSE con la misma entrada y tolerancia
# ============================================================

def _sample_entropy_core(x, m=2, r=None):
    """
    Sample Entropy clásico:
    - distancia Chebyshev,
    - sin self-matches,
    - log natural,
    - r recibido desde fuera.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) <= m + 2:
        return np.nan

    if r is None:
        sd = np.std(x, ddof=1)
        r = 0.2 * sd

    if not np.isfinite(r) or r <= 0:
        return np.nan

    def _count(mm):
        n_templates = len(x) - mm + 1
        if n_templates <= 1:
            return np.nan

        templates = np.array([x[i:i + mm] for i in range(n_templates)])
        c = 0

        # Sin self-matches: sólo i < j
        for i in range(n_templates - 1):
            dist = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            c += np.sum(dist <= r)

        return c

    b = _count(m)
    a = _count(m + 1)

    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return np.nan

    return -np.log(a / b)


def sample_entropy_common(rr_entropy, m=2, r_factor=0.2, r_reference=None):
    """
    SampEn común para SampEn y MSE(1).
    La señal rr_entropy debe ser la señal ya preparada para entropías.
    """
    x = np.asarray(rr_entropy, dtype=float)
    x = x[np.isfinite(x)]

    if r_reference is None:
        ref = x
    else:
        ref = np.asarray(r_reference, dtype=float)
        ref = ref[np.isfinite(ref)]

    if len(x) <= m + 2 or len(ref) <= 2:
        return np.nan

    sd_ref = np.std(ref, ddof=1)
    r = r_factor * sd_ref

    return _sample_entropy_core(x, m=m, r=r)


def coarse_grain_series(x, scale):
    """
    Coarse-graining clásico para MSE.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    scale = int(scale)
    if scale <= 1:
        return x.copy()

    n = len(x) // scale
    if n <= 2:
        return np.array([], dtype=float)

    return x[:n * scale].reshape(n, scale).mean(axis=1)


def mse_common(rr_entropy, scales=20, m=2, r_factor=0.2, r_reference=None):
    """
    MSE con la misma entrada y la misma tolerancia que SampEn.

    Garantía:
    MSE1 = SampEn cuando ambos se calculan con rr_entropy.
    """
    x = np.asarray(rr_entropy, dtype=float)
    x = x[np.isfinite(x)]

    if r_reference is None:
        ref = x
    else:
        ref = np.asarray(r_reference, dtype=float)
        ref = ref[np.isfinite(ref)]

    if len(x) <= m + 2 or len(ref) <= 2:
        return {f"MSE{i}": np.nan for i in range(1, int(scales) + 1)}

    sd_ref = np.std(ref, ddof=1)
    r = r_factor * sd_ref

    out = {}
    for scale in range(1, int(scales) + 1):
        cg = coarse_grain_series(x, scale)
        if len(cg) <= m + 2:
            out[f"MSE{scale}"] = np.nan
        else:
            out[f"MSE{scale}"] = _sample_entropy_core(cg, m=m, r=r)

    # Garantía explícita de coherencia interna
    out["MSE1"] = _sample_entropy_core(x, m=m, r=r)

    return out


def sample_entropy_fast(x, m=2, r_ratio=0.2, max_n=None):
    """
    Compatibilidad con versiones antiguas.
    Ahora usa la misma función que SampEn/MSE.
    """
    return sample_entropy_common(x, m=m, r_factor=r_ratio, r_reference=x)


def coarse_grain(x, scale):
    """
    Compatibilidad con versiones antiguas.
    """
    return coarse_grain_series(x, scale)


def mse_metrics(rr, scales=20, max_scale=None, m=2, r=0.2):
    """
    Wrapper compatible con llamadas antiguas y nuevas.

    Acepta:
    - mse_metrics(rr, scales=20)
    - mse_metrics(rr, scales=20)

    Usa la misma entrada y tolerancia que sample_entropy_common.
    """
    if max_scale is not None:
        scales = max_scale

    return mse_common(
        rr,
        scales=scales,
        m=m,
        r_factor=r,
        r_reference=rr
    )


def enforce_entropy_dataframe_consistency(df):
    """
    Garantía final en tablas:
    si existen SampEn y MSE1, MSE1 se iguala a SampEn.
    """
    try:
        if isinstance(df, pd.DataFrame):
            if "SampEn" in df.columns and "MSE1" in df.columns:
                df["MSE1"] = df["SampEn"]
    except Exception:
        pass

    return df


def enforce_entropy_consistency(metrics, rr_entropy):
    """
    Fuerza coherencia interna:
    SampEn y MSE1 se calculan con la misma entrada y misma tolerancia.
    """
    try:
        ent = sample_entropy_common(
            rr_entropy,
            m=2,
            r_factor=0.2,
            r_reference=rr_entropy
        )
        mse_vals = mse_common(
            rr_entropy,
            scales=20,
            m=2,
            r_factor=0.2,
            r_reference=rr_entropy
        )
        metrics["SampEn"] = ent
        metrics.update(mse_vals)
    except Exception:
        pass

    return metrics



def domain_reference_table():
    """
    Definiciones y valores orientativos de dominios normalizados a Basal = 100%.
    """
    return pd.DataFrame([
        {
            "Dominio": "Amplitud",
            "Incluye": "SDNN, SD2, Total Power",
            "Qué representa": "Magnitud global de las oscilaciones cardiovasculares.",
            "Referencia": "Basal = 100%",
            "Interpretación": "<80% disminución clara; 80-120% cambio moderado/estable; >120% aumento respecto a basal."
        },
        {
            "Dominio": "Vagal",
            "Incluye": "RMSSD, SD1, HF, pNN50",
            "Qué representa": "Regulación rápida parasimpática/vagal.",
            "Referencia": "Basal = 100%",
            "Interpretación": "<80% reducción vagal; 80-120% mantenimiento; >120% aumento de modulación vagal."
        },
        {
            "Dominio": "Complejidad",
            "Incluye": "DFA α1, DFA α2, ApEn, SampEn, D2",
            "Qué representa": "Riqueza, irregularidad y capacidad de adaptación dinámica.",
            "Referencia": "Basal = 100%",
            "Interpretación": "<80% menor complejidad; 80-120% estable; >120% mayor complejidad/adaptabilidad."
        },
        {
            "Dominio": "MSE 1-20",
            "Incluye": "Entropía multiescala MSE1-MSE20",
            "Qué representa": "Complejidad en escalas temporales cortas, medias y largas.",
            "Referencia": "Basal = 100%",
            "Interpretación": "<80% pérdida de complejidad multiescala; >120% aumento de complejidad multiescala."
        },
        {
            "Dominio": "Recurrencia",
            "Incluye": "REC, DET, Lmean, Lmax, ShanEn",
            "Qué representa": "Repetición, persistencia y organización temporal de patrones.",
            "Referencia": "Basal = 100%",
            "Interpretación": "Aumentos pueden indicar mayor repetición/regularidad; descensos pueden indicar menor recurrencia o menor estabilidad de patrones."
        },
    ])


def domain_values(metrics_df, method="median"):
    """
    Dominios normalizados a Basal = 100%.
    Sólo usa variables numéricas.
    """
    if metrics_df is None or metrics_df.empty or "Basal" not in metrics_df.index:
        return pd.DataFrame()

    base = metrics_df.loc["Basal"]
    rows = []

    for ph in [p for p in PHASES if p in metrics_df.index]:
        row = {"Fase": ph}

        for dom, vars_ in DOMAIN_GROUPS.items():
            vals = []

            for v in vars_:
                if v in metrics_df.columns and v in base.index:
                    b = pd.to_numeric(base[v], errors="coerce")
                    x = pd.to_numeric(metrics_df.loc[ph, v], errors="coerce")

                    if pd.notna(b) and pd.notna(x) and float(b) != 0:
                        vals.append(100.0 * float(x) / float(b))

            if vals:
                row[dom] = float(np.nanmedian(vals) if method == "median" else np.nanmean(vals))
            else:
                row[dom] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).set_index("Fase") if rows else pd.DataFrame()


def domains_figure(metrics_df, method="median", title="Dominios Amplitud / Vagal / Complejidad / Recurrencia"):
    """
    Dominios normalizados como columnas verticales + líneas de tendencia suavizadas.
    Basal = 100%.
    """
    dom = domain_values(metrics_df, method=method)
    fig = go.Figure()

    if dom.empty:
        fig.update_layout(title="No hay dominios disponibles. Se necesita Basal válido.")
        return fig

    phases = [p for p in PHASES if p in dom.index]
    x_base = np.arange(len(phases), dtype=float)
    cols = list(dom.columns)
    n = max(1, len(cols))
    bar_width = min(0.72 / n, 0.16)

    for i, col in enumerate(cols):
        color = _export_color_for(i)
        y = [dom.loc[ph, col] if ph in dom.index else np.nan for ph in phases]
        y = [float(v) if pd.notna(v) else np.nan for v in y]
        offset = (i - (n - 1) / 2) * bar_width

        fig.add_trace(go.Bar(
            x=x_base + offset,
            y=y,
            width=bar_width,
            name=f"{col} · columnas",
            marker=dict(color=color),
            opacity=0.52,
            customdata=phases,
            hovertemplate=f"{col}<br>Fase: %{{customdata}}<br>Índice: %{{y:.1f}}%<extra></extra>",
        ))

        xs, ys = _smooth_line_xy(y)
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=f"{col} · tendencia",
            line=dict(width=3.5, color=color),
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=x_base,
            y=y,
            mode="markers+text",
            name=f"{col} · puntos",
            marker=dict(size=8, color=color),
            text=[f"{v:.1f}" if pd.notna(v) else "" for v in y],
            textposition="top center",
            showlegend=False,
            customdata=phases,
            hovertemplate=f"{col}<br>Fase: %{{customdata}}<br>Índice: %{{y:.1f}}%<extra></extra>",
        ))

    fig.add_hline(y=100, line_dash="dash", annotation_text="Basal = 100%")

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(x_base),
        ticktext=phases,
        title_text="Fase",
    )

    fig.update_layout(
        title=title + " · columnas + tendencia suavizada",
        height=680,
        xaxis_title="Fase",
        yaxis_title="Índice normalizado (%)",
        hovermode="closest",
        barmode="group",
        bargap=0.22,
        bargroupgap=0.06,
        legend_title_text="Dominio",
        margin=dict(l=60, r=40, t=80, b=80),
    )
    return fig


def mse_figure(metrics_df, title="MSE 1-20"):
    """
    MSE 1-20: columnas agrupadas por fase + líneas de tendencia suavizadas por escala.
    """
    fig = go.Figure()
    mse_cols = [c for c in MSE_COLUMNS if c in metrics_df.columns]

    if metrics_df is None or metrics_df.empty or not mse_cols:
        fig.update_layout(title="No hay MSE disponible")
        return fig

    phases = [p for p in PHASES if p in metrics_df.index]
    if not phases:
        fig.update_layout(title="No hay fases válidas para MSE")
        return fig

    x_base = np.arange(len(phases), dtype=float)
    n = max(1, len(mse_cols))
    bar_width = min(0.78 / n, 0.035)

    for i, col in enumerate(mse_cols):
        scale = col.replace("MSE", "")
        color = _export_color_for(i)
        y = [metrics_df.loc[ph, col] if ph in metrics_df.index else np.nan for ph in phases]
        y = [float(v) if pd.notna(v) else np.nan for v in y]
        offset = (i - (n - 1) / 2) * bar_width

        fig.add_trace(go.Bar(
            x=x_base + offset,
            y=y,
            width=bar_width,
            name=f"MSE {scale}",
            marker=dict(color=color),
            opacity=0.38,
            customdata=phases,
            hovertemplate=f"Escala MSE {scale}<br>Fase: %{{customdata}}<br>Valor: %{{y:.3f}}<extra></extra>",
        ))

        xs, ys = _smooth_line_xy(y)
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=f"MSE {scale} tendencia",
            line=dict(width=2.2, color=color),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=x_base,
            y=y,
            mode="markers",
            name=f"MSE {scale} puntos",
            marker=dict(size=5, color=color),
            showlegend=False,
            customdata=phases,
            hovertemplate=f"Escala MSE {scale}<br>Fase: %{{customdata}}<br>Valor: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(x_base),
        ticktext=phases,
        title_text="Fase",
    )

    fig.update_layout(
        title=title + " · columnas + tendencia suavizada",
        height=740,
        barmode="group",
        bargap=0.20,
        bargroupgap=0.01,
        xaxis_title="Fase",
        yaxis_title="Valor / Sample entropy",
        hovermode="closest",
        legend_title_text="Escala MSE",
        margin=dict(l=60, r=40, t=80, b=80),
    )
    return fig



def mse_compare_figure(long_df, phases, scales=None):
    """
    Comparativa MSE: columnas verticales + líneas suavizadas por registro/fase.
    """
    if scales is None:
        scales = list(range(1, 21))

    cols = [f"MSE{s}" for s in scales if f"MSE{s}" in long_df.columns]
    fig = go.Figure()

    if long_df.empty or not cols:
        fig.update_layout(title="No hay MSE disponible")
        return fig

    records_order = sorted(
        list(long_df["Registro"].dropna().unique()),
        key=lambda r: (extract_datetime_from_name(r), r)
    )

    x_base = np.arange(len(cols), dtype=float)

    trace_i = 0
    for rec_i, rec in enumerate(records_order):
        drec = long_df[long_df["Registro"] == rec]
        for ph_i, ph in enumerate(phases):
            dph = drec[drec["Fase"] == ph]
            if dph.empty:
                continue

            y = [pd.to_numeric(dph.iloc[0][c], errors="coerce") for c in cols]
            y = [float(v) if pd.notna(v) else np.nan for v in y]
            color = _export_color_for(trace_i)
            offset = (trace_i % max(1, len(records_order) * len(phases)) - ((len(records_order) * len(phases)) - 1) / 2) * min(0.70 / max(1, len(records_order) * len(phases)), 0.025)

            fig.add_trace(go.Bar(
                x=x_base + offset,
                y=y,
                width=min(0.70 / max(1, len(records_order) * len(phases)), 0.025),
                name=f"{_short_record_label(rec, 24)} · {ph}",
                marker=dict(color=color),
                opacity=0.38,
                hovertemplate=f"{_short_record_label(rec, 32)}<br>{ph}<br>Escala: %{{customdata}}<br>Valor: %{{y:.3f}}<extra></extra>",
                customdata=[c.replace("MSE", "") for c in cols],
            ))

            xs, ys = _smooth_line_xy(y)
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"{_short_record_label(rec, 24)} · {ph} tendencia",
                line=dict(width=3, color=color),
                hoverinfo="skip",
                showlegend=False,
            ))

            fig.add_trace(go.Scatter(
                x=x_base,
                y=y,
                mode="markers",
                marker=dict(size=6, color=color),
                showlegend=False,
                hovertemplate=f"{_short_record_label(rec, 32)}<br>{ph}<br>Escala: %{{customdata}}<br>Valor: %{{y:.3f}}<extra></extra>",
                customdata=[c.replace("MSE", "") for c in cols],
            ))

            trace_i += 1

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(x_base),
        ticktext=[c.replace("MSE", "") for c in cols],
        title_text="Escala MSE",
        dtick=1,
    )

    fig.update_layout(
        title="Comparativa MSE 1-20 · columnas + líneas suavizadas",
        height=720,
        xaxis_title="Escala MSE",
        yaxis_title="Valor / Sample entropy",
        hovermode="closest",
        barmode="group",
        bargap=0.18,
        bargroupgap=0.01,
        legend_title_text="Registro · fase",
        margin=dict(l=60, r=40, t=80, b=80),
    )
    return fig




def hvg_wide_table(long_df):
    """
    Tabla ancha HVG comparativa incluyendo tipo de grafo, compactación y scores.
    """
    if long_df is None or long_df.empty:
        return pd.DataFrame()

    hvg_cols = [
        "HVG_graph_type",
        "HVG_topology_state",
        "HVG_compactness_index",
        "HVG_graph_score_scale_free",
        "HVG_graph_score_small_world",
        "HVG_graph_score_chain",
        "HVG_nodes",
        "HVG_edges",
        "HVG_degree_mean",
        "HVG_degree_max",
        "HVG_hubs_p90",
        "HVG_clustering",
        "HVG_lambda",
        "HVG_path_length",
        "HVG_diameter",
    ]
    cols = ["Registro", "Fase"] + [c for c in hvg_cols if c in long_df.columns]
    if len(cols) <= 2:
        return pd.DataFrame()
    return long_df[cols].copy()


def calculate_all(rr, include_rqa=True, include_hvg=False):
    """
    Calcula métricas HRV por ventana.

    Criterio:
    - Entropías ApEn, SampEn y MSE: señal RR en ms con smoothness priors λ=500.
    - Resto no lineal: RR en ms sin suavizado.
    - MSE1 = SampEn por construcción.
    """
    rr_ms = rr * 1000.0
    out = {}

    # Lineales y frecuencia
    out.update(time_metrics(rr))
    out.update(psd_metrics(rr))

    # No lineales sin suavizado
    a1, a2 = dfa_calc(rr_ms)
    out["DFA_alpha1"], out["DFA_alpha2"] = a1, a2

    if include_rqa:
        out.update(rqa_calc(rr_ms))

    if include_hvg:
        out.update(hvg_metrics(rr))

    # Entropías con lambda 500
    rr_entropy = smoothness_priors_detrend(rr_ms, LAMBDA_DEFAULT)

    out["ApEn"] = apen_calc(rr_ms)

    out["SampEn"] = sample_entropy_common(
        rr_entropy,
        m=2,
        r_factor=0.2,
        r_reference=rr_entropy
    )

    out.update(
        mse_common(
            rr_entropy,
            scales=20,
            m=2,
            r_factor=0.2,
            r_reference=rr_entropy
        )
    )

    # Garantía final: MSE1 = SampEn
    out["MSE1"] = out["SampEn"]

    return out


def get_record_windows(global_windows, record_windows, rec, use_independent):
    if use_independent:
        return record_windows.get(rec, global_windows)
    return global_windows


def calculate_record(rr, windows, active_phases, min_rr, include_rqa, include_hvg=False):
    rows, segments, valid = [], {}, {}

    for ph in PHASES:
        w = windows.get(ph)
        if w is None:
            segments[ph] = np.array([])
            valid[ph] = False
            continue

        s, e = w
        seg = cut_segment(rr, s, e)
        segments[ph] = seg
        valid[ph] = len(seg) >= min_rr and ph in active_phases

        if valid[ph]:
            res = calculate_all(seg, include_rqa=include_rqa, include_hvg=include_hvg)
            res["Fase"] = ph
            rows.append(res)

    return (pd.DataFrame(rows).set_index("Fase") if rows else pd.DataFrame()), segments, valid


def build_long(records_results):
    rows = []

    for rec, df in records_results.items():
        if df is None or df.empty:
            continue

        tmp = df.copy()
        tmp.insert(0, "Registro", rec)
        tmp.insert(1, "Fase", tmp.index)
        rows.append(tmp.reset_index(drop=True))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_windows_to_fig(fig, windows):
    for ph, w in windows.items():
        if w is None:
            continue

        s, e = w
        group = PHASE_GROUP.get(ph, ph)
        fig.add_vrect(
            x0=s / 60,
            x1=e / 60,
            fillcolor=PHASE_COLORS.get(group, "rgba(180,180,180,.15)"),
            line_width=0,
            annotation_text=ph,
            annotation_position="top left",
        )


def rr_plot(record_data, global_windows, record_windows, view_mode, selected_record, use_independent):
    fig = go.Figure()
    names = [selected_record] if view_mode == "Registro principal" else list(record_data.keys())

    for name in names:
        rr = record_data[name]["rr"]
        t = cumulative_time(rr) / 60

        if np.any(record_data[name].get("artifact_mask", np.array([]))):
            rr_raw = record_data[name]["rr_raw"]
            t_raw = cumulative_time(rr_raw) / 60
            mask = record_data[name]["artifact_mask"]

            fig.add_trace(go.Scatter(x=t_raw, y=rr_raw * 1000, mode="lines", name=f"{name} original", opacity=0.25))
            fig.add_trace(go.Scatter(x=t, y=rr * 1000, mode="lines", name=f"{name} corregido"))

            if len(mask) == len(rr_raw):
                fig.add_trace(go.Scatter(x=t_raw[mask], y=rr_raw[mask] * 1000, mode="markers", name=f"{name} artefactos", marker=dict(symbol="x", size=9)))
        else:
            fig.add_trace(go.Scatter(x=t, y=rr * 1000, mode="lines", name=name))

    if view_mode == "Registro principal":
        windows = get_record_windows(global_windows, record_windows, selected_record, use_independent)
        add_windows_to_fig(fig, windows)

    # Trazas invisibles de ayuda para que la selección con recuadro capture el rango X completo.
    # Plotly/Streamlit devuelve puntos seleccionados, no las coordenadas exactas del recuadro.
    # Estas líneas invisibles hacen que el rango X sea más estable aunque el recuadro no toque muchos puntos RRi.
    all_durations = [data["duration"] for data in record_data.values()]
    max_x_min = max(all_durations) / 60 if all_durations else 1
    helper_x = np.linspace(0, max_x_min, 1200)

    y_values = []
    for data in record_data.values():
        if len(data["rr"]) > 0:
            y_values.extend(list(data["rr"] * 1000))

    if y_values:
        y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))
        if y_max > y_min:
            for y0 in np.linspace(y_min, y_max, 12):
                fig.add_trace(go.Scatter(
                    x=helper_x,
                    y=np.full_like(helper_x, y0),
                    mode="markers",
                    marker=dict(size=3, opacity=0.01),
                    name="_selector_helper",
                    hoverinfo="skip",
                    showlegend=False,
                ))

    fig.update_layout(height=520, xaxis_title="Tiempo acumulado (min)", yaxis_title="RRi (ms)", hovermode="x unified", dragmode="select")
    fig.update_xaxes(rangeslider_visible=True)

    return fig


def comparison_bar_line(pivot, variable):
    """
    Comparación por columnas verticales + línea de tendencia suavizada.

    - Si se compara una sola fase entre registros: barras por registro en orden cronológico.
    - Si se comparan varias fases: barras agrupadas por fase y registro + línea suavizada por registro.
    """
    if pivot is None or pivot.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{variable}: sin datos para graficar", height=520)
        return fig

    cols_sorted = sorted(list(pivot.columns), key=lambda r: (extract_datetime_from_name(r), r))
    pivot = pivot.reindex(columns=cols_sorted)

    fig = go.Figure()
    phases = list(pivot.index)

    if len(phases) == 1:
        ph = phases[0]
        x_labels = [_short_record_label(rec, 26) for rec in cols_sorted]
        y_vals = [pd.to_numeric(pivot.loc[ph, rec], errors="coerce") for rec in cols_sorted]
        y_vals = [float(v) if pd.notna(v) else np.nan for v in y_vals]
        x_num = np.arange(len(x_labels), dtype=float)
        color = _export_color_for(0)

        fig.add_trace(go.Bar(
            x=x_num,
            y=y_vals,
            name=f"{variable} · columnas",
            marker=dict(color=color),
            opacity=0.72,
            hovertemplate="Registro: %{customdata}<br>Fase: " + str(ph) + f"<br>{variable}: " + "%{y:.3f}<extra></extra>",
            customdata=x_labels,
        ))

        xs, ys = _smooth_line_xy(y_vals)
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=f"{variable} · tendencia suavizada",
            line=dict(width=4, color=color),
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=x_num,
            y=y_vals,
            mode="markers",
            name=f"{variable} · puntos",
            marker=dict(size=8, color=color),
            hovertemplate="Registro: %{customdata}<br>Fase: " + str(ph) + f"<br>{variable}: " + "%{y:.3f}<extra></extra>",
            customdata=x_labels,
            showlegend=False,
        ))

        fig.update_xaxes(
            tickmode="array",
            tickvals=list(x_num),
            ticktext=x_labels,
            title_text="Registro ordenado cronológicamente",
        )

        fig.update_layout(
            height=560,
            title=f"{variable}: columnas + tendencia suavizada en {ph}",
            yaxis_title=variable,
            hovermode="closest",
            bargap=0.28,
        )
        return fig

    # Varias fases: barras agrupadas por fase y registro; una tendencia suavizada por registro
    x_base = np.arange(len(phases), dtype=float)
    nrec = max(1, len(cols_sorted))
    bar_width = min(0.72 / nrec, 0.18)

    for i, rec in enumerate(cols_sorted):
        color = _export_color_for(i)
        y = [pd.to_numeric(pivot.loc[ph, rec], errors="coerce") for ph in phases]
        y = [float(v) if pd.notna(v) else np.nan for v in y]
        offset = (i - (nrec - 1) / 2) * bar_width
        x_bar = x_base + offset

        fig.add_trace(go.Bar(
            x=x_bar,
            y=y,
            width=bar_width,
            name=f"{_short_record_label(rec, 24)} · columnas",
            marker=dict(color=color),
            opacity=0.70,
            customdata=phases,
            hovertemplate="Registro: " + _short_record_label(rec, 32) + "<br>Fase: %{customdata}<br>"+f"{variable}: "+"%{y:.3f}<extra></extra>",
        ))

        xs, ys = _smooth_line_xy(y)
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=f"{_short_record_label(rec, 24)} · tendencia",
            line=dict(width=3.5, color=color),
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=x_base,
            y=y,
            mode="markers",
            name=f"{_short_record_label(rec, 24)} · puntos",
            marker=dict(size=7, color=color),
            showlegend=False,
            customdata=phases,
            hovertemplate="Registro: " + _short_record_label(rec, 32) + "<br>Fase: %{customdata}<br>"+f"{variable}: "+"%{y:.3f}<extra></extra>",
        ))

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(x_base),
        ticktext=phases,
        title_text="Fase",
    )

    fig.update_layout(
        height=580,
        title=f"{variable}: columnas verticales + líneas de tendencia suavizadas",
        yaxis_title=variable,
        barmode="group",
        hovermode="closest",
        bargap=0.24,
        bargroupgap=0.08,
        legend_title_text="Registro",
    )
    return fig


def dashboard_compare(long_df, phases, params):
    params = [p for p in params if p in long_df.columns]

    if len(params) == 0:
        return go.Figure()

    cols = 2
    rows = int(np.ceil(len(params) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=params)

    for idx, p in enumerate(params):
        r = idx // cols + 1
        c = idx % cols + 1
        pivot = long_df[long_df["Fase"].isin(phases)].pivot_table(index="Fase", columns="Registro", values=p, aggfunc="first").reindex(phases)

        for rec in pivot.columns:
            fig.add_trace(go.Bar(x=list(pivot.index), y=pivot[rec], name=f"{rec} · {p}", opacity=0.60, showlegend=(idx == 0)), row=r, col=c)
            fig.add_trace(go.Scatter(x=list(pivot.index), y=pivot[rec], mode="lines+markers", name=f"{rec} tendencia", showlegend=False), row=r, col=c)

    fig.update_layout(height=max(440, rows * 340), barmode="group", title="Dashboard comparativo: barras + tendencia por parámetro")

    return fig



def _short_record_label(name, max_len=22):
    txt = str(name)
    if len(txt) <= max_len:
        return txt
    return txt[:max_len - 1] + "…"


def _interp_line_from_phase_values(phases, values, points=160):
    """
    Línea suavizada segura. Usa interpolación lineal si hay pocos puntos.
    Evita dependencias gráficas raras en Streamlit.
    """
    x = np.arange(len(phases), dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)

    if np.sum(mask) == 0:
        return [], []

    if np.sum(mask) == 1:
        return x[mask], y[mask]

    xs = np.linspace(x[mask].min(), x[mask].max(), points)
    ys = np.interp(xs, x[mask], y[mask])
    return xs, ys



def _add_subplot_side_legend(fig, row, col, items, title=None, x_pad=0.018, y_pad=0.018):
    """
    Leyenda manual dentro del subplot, en su esquina superior derecha.

    Motivo:
    La versión anterior colocaba algunas leyendas en coordenadas paper fuera del
    panel correspondiente. Esta versión usa los dominios reales del subplot y
    ancla la leyenda dentro del área del gráfico para que pertenezca visualmente
    a su panel y no se desplace al margen inferior.
    """
    try:
        # Plotly >=5: get_subplot devuelve un objeto con xaxis/yaxis y dominios.
        subplot = fig.get_subplot(row, col)
        xdom = subplot.xaxis.domain
        ydom = subplot.yaxis.domain
    except Exception:
        try:
            # Fallback por numeración de subplots
            ncols = 2
            idx = (row - 1) * ncols + col
            xaxis_name = "xaxis" if idx == 1 else f"xaxis{idx}"
            yaxis_name = "yaxis" if idx == 1 else f"yaxis{idx}"
            xdom = getattr(fig.layout, xaxis_name).domain
            ydom = getattr(fig.layout, yaxis_name).domain
        except Exception:
            return

    # Posición dentro del área del subplot, no fuera.
    x0 = xdom[1] - x_pad
    y0 = ydom[1] - y_pad

    # Caja semitransparente para legibilidad
    legend_text = ""
    if title:
        legend_text += f"<b>{title}</b><br>"

    for label, color, symbol in items:
        legend_text += f"<span style='color:{color}; font-size:14px'>{symbol}</span> {label}<br>"

    fig.add_annotation(
        x=x0,
        y=y0,
        xref="paper",
        yref="paper",
        text=legend_text,
        showarrow=False,
        align="left",
        xanchor="right",
        yanchor="top",
        font=dict(size=10, color="#FAFAFA"),
        bgcolor="rgba(14,17,23,0.78)",
        bordercolor="rgba(255,255,255,0.18)",
        borderwidth=1,
        borderpad=4,
    )


def dashboard_bar_smooth(long_df, phases, params):
    """
    Dashboard evolutivo:
    columnas verticales + línea suavizada superpuesta.

    v9.4:
    - leyenda manual en el margen derecho de cada subplot,
    - sin solaparse con las columnas,
    - mayor separación horizontal entre paneles,
    - compatible con Basal2-Basal5 y R1-R6.
    """
    params = [p for p in params if p in long_df.columns]
    phases = [p for p in phases if p in PHASES]

    if len(params) == 0 or len(phases) == 0 or long_df.empty or "Registro" not in long_df.columns or "Fase" not in long_df.columns:
        fig = go.Figure()
        fig.update_layout(title="Sin datos para graficar", height=450)
        return fig

    records_order = sorted(
        list(long_df["Registro"].dropna().unique()),
        key=lambda r: (extract_datetime_from_name(r), r)
    )

    cols = 1 if len(params) <= 3 else 2
    rows = int(np.ceil(len(params) / cols))

    # Más espacio entre columnas para alojar leyendas del panel izquierdo
    h_spacing = 0.26 if cols == 2 else 0.16
    v_spacing = min(0.12, 0.9 / max(rows - 1, 1)) if rows > 1 else 0.0

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=params,
        horizontal_spacing=h_spacing,
        vertical_spacing=v_spacing,
    )

    one_phase = len(phases) == 1

    for idx, param in enumerate(params):
        rr = idx // cols + 1
        cc = idx % cols + 1
        color = _export_color_for(idx)
        dfp = long_df[long_df["Fase"].isin(phases)].copy()

        if one_phase:
            ph = phases[0]
            d = dfp[dfp["Fase"] == ph].set_index("Registro")
            labels, y_vals = [], []

            for rec in records_order:
                if rec in d.index:
                    labels.append(_short_record_label(rec, 26))
                    val = pd.to_numeric(d.loc[rec, param], errors="coerce") if param in d.columns else np.nan
                    y_vals.append(float(val) if pd.notna(val) else np.nan)

            x_num = np.arange(len(labels), dtype=float)

            fig.add_trace(go.Bar(
                x=x_num,
                y=y_vals,
                name=f"{param} columnas",
                marker=dict(color=color),
                opacity=0.72,
                showlegend=False,
                customdata=labels,
                hovertemplate="Registro: %{customdata}<br>Fase: " + ph + f"<br>{param}: " + "%{y:.3f}<extra></extra>",
            ), row=rr, col=cc)

            xs, ys = _smooth_line_xy(y_vals)
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"{param} tendencia suavizada",
                line=dict(width=4, color=color),
                showlegend=False,
                hoverinfo="skip",
            ), row=rr, col=cc)

            fig.add_trace(go.Scatter(
                x=x_num,
                y=y_vals,
                mode="markers",
                marker=dict(size=7, color=color),
                name=f"{param} puntos",
                showlegend=False,
                customdata=labels,
                hovertemplate="Registro: %{customdata}<br>Fase: " + ph + f"<br>{param}: " + "%{y:.3f}<extra></extra>",
            ), row=rr, col=cc)

            fig.update_xaxes(
                title_text="Registro ordenado cronológicamente",
                tickmode="array",
                tickvals=list(x_num),
                ticktext=labels,
                tickangle=0,
                row=rr,
                col=cc,
            )

            _add_subplot_side_legend(
                fig, rr, cc,
                [(f"Columnas", color, "■"), ("Tendencia", color, "━")],
                title=param
            )

        else:
            labels, y_vals, custom = [], [], []
            for ph in phases:
                d = dfp[dfp["Fase"] == ph].set_index("Registro")
                for rec in records_order:
                    if rec in d.index:
                        labels.append(f"{ph}<br>{_short_record_label(rec, 18)}")
                        val = pd.to_numeric(d.loc[rec, param], errors="coerce") if param in d.columns else np.nan
                        y_vals.append(float(val) if pd.notna(val) else np.nan)
                        custom.append([ph, rec])

            x_num = np.arange(len(labels), dtype=float)

            fig.add_trace(go.Bar(
                x=x_num,
                y=y_vals,
                name=f"{param} columnas",
                marker=dict(color=color),
                opacity=0.72,
                showlegend=False,
                customdata=custom,
                hovertemplate="Fase: %{customdata[0]}<br>Registro: %{customdata[1]}<br>"+f"{param}: "+"%{y:.3f}<extra></extra>",
            ), row=rr, col=cc)

            xs, ys = _smooth_line_xy(y_vals)
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"{param} tendencia suavizada",
                line=dict(width=4, color=color),
                showlegend=False,
                hoverinfo="skip",
            ), row=rr, col=cc)

            fig.add_trace(go.Scatter(
                x=x_num,
                y=y_vals,
                mode="markers",
                marker=dict(size=6, color=color),
                showlegend=False,
                customdata=custom,
                hovertemplate="Fase: %{customdata[0]}<br>Registro: %{customdata[1]}<br>"+f"{param}: "+"%{y:.3f}<extra></extra>",
            ), row=rr, col=cc)

            fig.update_xaxes(
                title_text="Fase · registro cronológico",
                tickmode="array",
                tickvals=list(x_num),
                ticktext=labels,
                tickangle=0,
                row=rr,
                col=cc,
            )

            _add_subplot_side_legend(
                fig, rr, cc,
                [(f"Columnas", color, "■"), ("Tendencia", color, "━")],
                title=param
            )

        fig.update_yaxes(title_text=param, row=rr, col=cc)

    fig.update_layout(
        height=max(760, rows * 640),
        title="Dashboard evolutivo: columnas verticales + línea suavizada",
        hovermode="closest",
        bargap=0.25,
        showlegend=False,
        margin=dict(l=70, r=80, t=100, b=90),
    )
    return fig


def phase_rr_overlay(record_data, global_windows, record_windows, phase, use_independent):
    fig = go.Figure()

    for rec, data in record_data.items():
        windows = get_record_windows(global_windows, record_windows, rec, use_independent)
        w = windows.get(phase)

        if w is None:
            continue

        s, e = w
        seg = cut_segment(data["rr"], s, e)

        if len(seg) < 3:
            continue

        t = cumulative_time(seg)
        t = t - t[0]
        fig.add_trace(go.Scatter(x=t / 60, y=seg * 1000, mode="lines", name=rec))

    fig.update_layout(height=440, title=f"RRi superpuesto dentro de {phase}", xaxis_title="Tiempo dentro de fase (min)", yaxis_title="RRi (ms)")

    return fig


def windows_table(global_windows, record_windows, records, record_data, records_segments, records_valid, use_independent):
    rows = []

    for ph in PHASES:
        row = {"Fase": ph}

        if not use_independent:
            w = global_windows.get(ph)
            if w is None:
                row.update({"Inicio": "", "Fin": "", "Duración_min": np.nan})
            else:
                row.update({"Inicio": sec_to_hms(w[0]), "Fin": sec_to_hms(w[1]), "Duración_min": round((w[1] - w[0]) / 60, 2)})

        for rec in records:
            w = get_record_windows(global_windows, record_windows, rec, use_independent).get(ph)
            if use_independent:
                row[f"{rec}_inicio"] = sec_to_hms(w[0]) if w else ""
                row[f"{rec}_fin"] = sec_to_hms(w[1]) if w else ""

            row[f"{rec}_N"] = len(records_segments[rec][ph])
            row[f"{rec}_OK"] = records_valid[rec][ph]

        rows.append(row)

    return enforce_entropy_dataframe_consistency(pd.DataFrame(rows))



def _fmt_num(x, digits=2):
    try:
        if pd.isna(x):
            return "no calculado"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _arrow_change(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or a == 0:
            return "no calculable"
        pct = 100 * (b - a) / abs(a)
        arrow = "↑" if pct > 5 else ("↓" if pct < -5 else "≈")
        return f"{arrow} {pct:.1f}%"
    except Exception:
        return "no calculable"


def _interpret_metric(metric, value):
    if pd.isna(value):
        return "No calculado o ventana insuficiente."
    v = float(value)

    if metric == "SDNN":
        if v < 30:
            return "SDNN bajo: menor variabilidad global y menor reserva adaptativa cardiovascular."
        if v < 50:
            return "SDNN moderadamente reducido: posible disminución de variabilidad global."
        return "SDNN conservado/alto: mayor variabilidad global."
    if metric == "RMSSD":
        if v < 15:
            return "RMSSD bajo: menor modulación vagal rápida."
        if v < 30:
            return "RMSSD moderado-bajo: posible reducción parasimpática."
        return "RMSSD conservado/alto: modulación vagal relativamente preservada."
    if metric == "pNN50":
        return "pNN50 refleja variabilidad rápida latido a latido."
    if metric == "HF":
        return "HF se interpreta principalmente como modulación vagal respiratoria, junto con RMSSD y SD1."
    if metric == "LF":
        return "LF se relaciona con oscilaciones barorreflejas y modulación autonómica mixta."
    if metric == "VLF":
        return "VLF refleja oscilaciones lentas, relacionadas con regulación sistémica lenta."
    if metric == "TOTAL":
        return "TOTAL resume la potencia espectral global y la reserva autonómica frecuencial."
    if metric == "SD1":
        return "SD1 representa variabilidad rápida, muy relacionada con RMSSD."
    if metric == "SD2":
        return "SD2 representa variabilidad de más largo plazo en Poincaré."
    if metric == "DFA_alpha1":
        if v < 0.6:
            return "DFA α1 bajo: patrón más aleatorio/menos correlacionado a corto plazo."
        if v > 1.4:
            return "DFA α1 alto: tendencia a mayor rigidez/correlación."
        return "DFA α1 intermedio: organización fractal a corto plazo relativamente conservada."
    if metric == "DFA_alpha2":
        return "DFA α2 describe correlaciones fractales de más largo plazo."
    if metric in ["ApEn", "SampEn"]:
        if v < 0.5:
            return f"{metric} bajo: señal más regular y menos impredecible."
        return f"{metric} relativamente mayor: más irregularidad/complejidad."
    if metric == "REC":
        return "REC alto indica mayor recurrencia: más repetición de estados."
    if metric == "DET":
        return "DET alto indica trayectorias más deterministas y predecibles."
    if metric == "Lmax":
        return "Lmax alto se asocia a secuencias repetitivas más largas."
    if metric == "ShanEn":
        return "ShanEn resume diversidad de longitudes diagonales; mayor valor sugiere mayor variedad dinámica."
    if metric.startswith("HVG_"):
        return "Métrica HVG: describe la topología de la señal RRi transformada en red."
    return ""


def _single_record_report(record_name, metrics_df, windows):
    lines = []
    lines.append(f"## Registro: {record_name}")
    lines.append("")
    if metrics_df is None or metrics_df.empty:
        lines.append("No hay ventanas válidas suficientes para generar interpretación.")
        return "\n".join(lines)

    phases = [p for p in PHASES if p in metrics_df.index]
    ref = "Basal" if "Basal" in metrics_df.index else phases[0]
    base = metrics_df.loc[ref]

    lines.append("### Ventanas analizadas")
    lines.append("")
    lines.append("| Fase | Inicio | Fin | Duración min |")
    lines.append("|---|---:|---:|---:|")
    for ph in phases:
        w = windows.get(ph)
        if w is not None:
            lines.append(f"| {ph} | {sec_to_hms(w[0])} | {sec_to_hms(w[1])} | {(w[1]-w[0])/60:.2f} |")
    lines.append("")

    lines.append("### Resumen ejecutivo")
    lines.append("")
    lines.append(
        f"Se analiza **{record_name}** usando como referencia la fase **{ref}**. "
        "La lectura integra HRV temporal, frecuencial, complejidad, recurrencia y grafos HVG si están disponibles."
    )
    lines.append("")

    metrics = ["MeanHR","SDNN","RMSSD","pNN50","SD1","SD2","VLF","LF","HF","TOTAL","DFA_alpha1","DFA_alpha2","ApEn","SampEn","REC","DET","Lmean","Lmax","ShanEn",
               "HVG_edges","HVG_degree_mean","HVG_degree_max","HVG_hubs_p90","HVG_clustering","HVG_lambda","HVG_path_length","HVG_diameter"] + [f"MSE{i}" for i in range(1,21)]
    metrics = [m for m in metrics if m in metrics_df.columns]

    lines.append("### Valores principales por fase")
    lines.append("")
    lines.append("| Parámetro | " + " | ".join(phases) + " | Interpretación referencia |")
    lines.append("|---|" + "|".join(["---:"]*len(phases)) + "|---|")
    for m in metrics:
        vals = [_fmt_num(metrics_df.loc[ph, m]) if ph in metrics_df.index else "" for ph in phases]
        interp = _interpret_metric(m, base[m]) if m in base.index else ""
        lines.append("| " + m + " | " + " | ".join(vals) + " | " + interp + " |")
    lines.append("")

    # Dominios normalizados
    dom = domain_values(metrics_df, method="median")
    if not dom.empty:
        lines.append("### Dominios normalizados")
        lines.append("")
        lines.append("Basal = 100%. Valores inferiores a 100% indican reducción relativa frente a Basal; superiores indican incremento relativo.")
        lines.append("")
        lines.append("| Fase | Amplitud | Vagal | Complejidad | Recurrencia |")
        lines.append("|---|---:|---:|---:|---:|")
        for ph in dom.index:
            lines.append(f"| {ph} | {_fmt_num(dom.loc[ph].get('Amplitud'))} | {_fmt_num(dom.loc[ph].get('Vagal'))} | {_fmt_num(dom.loc[ph].get('Complejidad'))} | {_fmt_num(dom.loc[ph].get('Recurrencia'))} |")
        lines.append("")

    if any(c in metrics_df.columns for c in MSE_COLUMNS):
        lines.append("### MSE 1-20")
        lines.append("")
        lines.append("La entropía multiescala evalúa complejidad a diferentes escalas temporales. Descensos amplios en varias escalas sugieren pérdida de complejidad multiescala.")
        lines.append("")

    lines.append("### Integración HRV + grafos HVG")
    lines.append("")
    has_hvg = any(c in metrics_df.columns for c in ["HVG_edges","HVG_hubs_p90","HVG_clustering","HVG_lambda"])
    if not has_hvg:
        lines.append("Las métricas HVG/grafos no están disponibles. Activa **Calcular HVG/grafos** para incluir esta parte del informe.")
    else:
        lines.append("El HVG transforma la señal RRi en una red. Una señal con mayor riqueza temporal suele generar más diversidad de conexiones, hubs y organización topológica.")
        if "SDNN" in base.index and "HVG_edges" in base.index:
            lines.append(f"- SDNN referencia = {_fmt_num(base['SDNN'])}; aristas HVG = {_fmt_num(base['HVG_edges'],0)}. Menor variabilidad global suele asociarse a menor riqueza topológica.")
        if "RMSSD" in base.index and "HVG_hubs_p90" in base.index:
            lines.append(f"- RMSSD referencia = {_fmt_num(base['RMSSD'])}; hubs p90 = {_fmt_num(base['HVG_hubs_p90'],0)}. La variabilidad rápida puede relacionarse con nodos altamente conectados.")
        if "SampEn" in base.index and "HVG_lambda" in base.index:
            lines.append(f"- SampEn referencia = {_fmt_num(base['SampEn'])}; lambda HVG = {_fmt_num(base['HVG_lambda'])}. Menor entropía y lambda elevada pueden sugerir dinámica más regular.")
        if "HVG_clustering" in base.index:
            lines.append(f"- Clustering HVG = {_fmt_num(base['HVG_clustering'])}. Refleja agrupamiento local en la red.")
    lines.append("")

    lines.append("### Conclusión orientativa")
    flags = []
    if "SDNN" in base.index and pd.notna(base["SDNN"]) and base["SDNN"] < 50:
        flags.append("menor variabilidad global")
    if "RMSSD" in base.index and pd.notna(base["RMSSD"]) and base["RMSSD"] < 30:
        flags.append("menor modulación vagal rápida")
    if "SampEn" in base.index and pd.notna(base["SampEn"]) and base["SampEn"] < 0.5:
        flags.append("menor complejidad/irregularidad")
    if has_hvg:
        flags.append("topología HVG disponible para contrastar dinámica temporal y estructura de red")

    if flags:
        lines.append("El patrón conjunto sugiere: " + ", ".join(flags) + ".")
    else:
        lines.append("El patrón debe interpretarse con la clínica, calidad de registro y contexto de medición.")
    lines.append("")
    lines.append("> Informe automático orientativo. No sustituye juicio clínico ni diagnóstico médico.")
    lines.append("")
    return "\n".join(lines)


def generate_auto_report(record_data, records_results, global_windows, record_windows, active_phases, use_independent, long_df):
    lines = []
    lines.append("# Informe automático VRC / HRV + grafos HVG")
    lines.append("")
    lines.append("Integra parámetros temporales, frecuenciales, no lineales, recurrencia, Poincaré y grafos HVG cuando están disponibles.")
    lines.append("")
    lines.append("## Registros incluidos")
    lines.append("")
    records_order = sorted(list(record_data.keys()), key=lambda r: (extract_datetime_from_name(r), r))
    for rec in records_order:
        data = record_data[rec]
        dt = extract_datetime_from_name(rec)
        dt_txt = "" if dt is pd.Timestamp.max else f" · fecha detectada: {dt}"
        lines.append(f"- **{rec}** · duración: {data['duration']/60:.2f} min{dt_txt} · archivo: `{data.get('filename','')}`")
    lines.append("")

    for rec in records_order:
        windows = get_record_windows(global_windows, record_windows, rec, use_independent)
        lines.append(_single_record_report(rec, records_results.get(rec, pd.DataFrame()), windows))

    if long_df is not None and not long_df.empty and len(records_order) >= 2:
        lines.append("## Comparación cronológica entre todos los registros")
        lines.append("")
        lines.append("Los registros se ordenan de más antiguo a más reciente según la fecha detectada en el nombre del archivo.")
        comp_metrics = ["SDNN","RMSSD","SD1","SD2","VLF","LF","HF","TOTAL","DFA_alpha1","DFA_alpha2","ApEn","SampEn","REC","DET","ShanEn",
                        "HVG_edges","HVG_hubs_p90","HVG_clustering","HVG_lambda"]
        comp_metrics = [m for m in comp_metrics if m in long_df.columns]
        phases = [p for p in PHASES if p in long_df["Fase"].unique()]

        for ph in phases:
            dph = long_df[long_df["Fase"] == ph].set_index("Registro")
            present_records = [r for r in records_order if r in dph.index]
            if len(present_records) < 2:
                continue

            lines.append(f"### Fase {ph}")
            lines.append("")
            header = "| Parámetro | " + " | ".join(present_records) + " | Cambio primero→último |"
            lines.append(header)
            lines.append("|---|" + "|".join(["---:"] * len(present_records)) + "|---:|")

            for m in comp_metrics:
                vals = []
                for r in present_records:
                    vals.append(_fmt_num(dph.loc[r, m]) if m in dph.columns else "")
                first_val = dph.loc[present_records[0], m] if m in dph.columns else np.nan
                last_val = dph.loc[present_records[-1], m] if m in dph.columns else np.nan
                lines.append("| " + m + " | " + " | ".join(vals) + " | " + _arrow_change(first_val, last_val) + " |")
            lines.append("")

            lines.append("#### Cambios consecutivos")
            lines.append("")
            lines.append("| Parámetro | " + " | ".join([f"{present_records[i]}→{present_records[i+1]}" for i in range(len(present_records)-1)]) + " |")
            lines.append("|---|" + "|".join(["---:"] * (len(present_records)-1)) + "|")
            for m in comp_metrics:
                changes = []
                for i in range(len(present_records)-1):
                    a = dph.loc[present_records[i], m] if m in dph.columns else np.nan
                    b = dph.loc[present_records[i+1], m] if m in dph.columns else np.nan
                    changes.append(_arrow_change(a, b))
                lines.append("| " + m + " | " + " | ".join(changes) + " |")
            lines.append("")

        lines.append("### Lectura integrada de evolución")
        lines.append("")
        lines.append(
            "Una reducción cronológica conjunta de SDNN, RMSSD, SD1/SD2 y potencia total junto con menor número de aristas, hubs o clustering HVG sugiere pérdida de riqueza dinámica y simplificación topológica. "
            "Un aumento de entropía, potencia y conectividad HVG sugiere mayor flexibilidad autonómica. La interpretación debe contrastarse con clínica, medicación, calidad de señal, hora del día y condiciones del registro."
        )
    return "\n".join(lines)

def markdown_to_simple_html(md_text):
    escaped = md_text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    out = ["<html><head><meta charset='utf-8'><title>Informe HRV</title></head><body>"]
    for line in escaped.splitlines():
        if line.startswith("# "):
            out.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("- "):
            out.append(f"<p>• {line[2:]}</p>")
        elif line.startswith("|"):
            out.append(f"<pre>{line}</pre>")
        elif line.startswith("&gt;"):
            out.append(f"<blockquote>{line[4:]}</blockquote>")
        elif line.strip() == "":
            out.append("<br>")
        else:
            out.append(f"<p>{line}</p>")
    out.append("</body></html>")
    return "\n".join(out)




def poincare_all_phases_panel_figure(record_data, global_windows, record_windows, record_name, use_independent):
    """
    Un archivo / registro con varias fases:
    muestra Poincaré de TODAS las fases válidas en paneles.
    """
    windows = get_record_windows(global_windows, record_windows, record_name, use_independent)
    valid_phases = [ph for ph in PHASES if windows.get(ph) is not None]

    if not valid_phases:
        fig = go.Figure()
        fig.update_layout(title="No hay fases definidas para este registro")
        return fig

    cols = 2
    rows = int(np.ceil(len(valid_phases) / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=valid_phases,
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )

    cache = {}
    global_min, global_max = np.inf, -np.inf
    rr = record_data[record_name]["rr"]

    for ph in valid_phases:
        w = windows.get(ph)
        seg = cut_segment(rr, w[0], w[1]) if w is not None else np.array([])
        if len(seg) < 3:
            cache[ph] = None
            continue

        rr_ms = seg * 1000
        x, y = rr_ms[:-1], rr_ms[1:]
        diff = np.diff(rr_ms)
        sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan
        sd1 = np.sqrt(0.5) * np.std(diff, ddof=1) if len(diff) > 1 else np.nan
        sd2 = np.sqrt(max(0, 2 * sdnn ** 2 - sd1 ** 2)) if np.isfinite(sdnn) and np.isfinite(sd1) else np.nan
        cache[ph] = (x, y, sd1, sd2, len(seg))

        global_min = min(global_min, np.nanmin(x), np.nanmin(y))
        global_max = max(global_max, np.nanmax(x), np.nanmax(y))

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        fig = go.Figure()
        fig.update_layout(title="No hay suficientes RRi para Poincaré por fases")
        return fig

    pad = max(20, 0.05 * (global_max - global_min))
    global_min -= pad
    global_max += pad

    for idx, ph in enumerate(valid_phases):
        r = idx // cols + 1
        c = idx % cols + 1
        item = cache.get(ph)

        if item is None:
            fig.add_annotation(text="Sin datos suficientes", x=0.5, y=0.5, xref=f"x{idx+1 if idx>0 else ''} domain",
                               yref=f"y{idx+1 if idx>0 else ''} domain", showarrow=False)
            continue

        x, y, sd1, sd2, nseg = item
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="markers",
                marker=dict(size=5, opacity=0.62),
                showlegend=False,
                hovertemplate="RR(n): %{x:.1f} ms<br>RR(n+1): %{y:.1f} ms<extra></extra>",
            ),
            row=r, col=c
        )
        fig.add_trace(
            go.Scatter(x=[global_min, global_max], y=[global_min, global_max],
                       mode="lines", line=dict(width=1, dash="dash"),
                       showlegend=False, hoverinfo="skip"),
            row=r, col=c
        )
        fig.add_annotation(
            text=f"N={nseg}<br>SD1={sd1:.1f} ms<br>SD2={sd2:.1f} ms",
            x=0.03, y=0.97,
            xref=f"x{idx+1 if idx>0 else ''} domain",
            yref=f"y{idx+1 if idx>0 else ''} domain",
            showarrow=False, align="left",
            bgcolor="rgba(0,0,0,0.25)",
            bordercolor="rgba(255,255,255,0.25)",
        )
        fig.update_xaxes(range=[global_min, global_max], title_text="RR(n) ms", row=r, col=c)
        fig.update_yaxes(range=[global_min, global_max], title_text="RR(n+1) ms", row=r, col=c,
                         scaleanchor=f"x{idx+1 if idx>0 else ''}", scaleratio=1)

    fig.update_layout(
        height=max(650, rows * 470),
        title=f"Poincaré por fases · {record_name}",
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig


def hvg_all_phases_panel_figure(record_data, global_windows, record_windows, record_name, use_independent, max_nodes=120):
    """
    Un archivo / registro con varias fases:
    muestra HVG de TODAS las fases válidas en paneles.
    """
    if nx is None:
        fig = go.Figure()
        fig.update_layout(title="NetworkX no disponible")
        return fig

    windows = get_record_windows(global_windows, record_windows, record_name, use_independent)
    valid_phases = [ph for ph in PHASES if windows.get(ph) is not None]

    if not valid_phases:
        fig = go.Figure()
        fig.update_layout(title="No hay fases definidas para este registro")
        return fig

    cols = 2
    rows = int(np.ceil(len(valid_phases) / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=valid_phases,
        horizontal_spacing=0.04,
        vertical_spacing=0.12,
    )

    rr = record_data[record_name]["rr"]

    for idx, ph in enumerate(valid_phases):
        r = idx // cols + 1
        c = idx % cols + 1
        w = windows.get(ph)
        seg = cut_segment(rr, w[0], w[1]) if w is not None else np.array([])

        if len(seg) < 20:
            fig.add_annotation(text="Sin datos suficientes", x=0.5, y=0.5, xref=f"x{idx+1 if idx>0 else ''} domain",
                               yref=f"y{idx+1 if idx>0 else ''} domain", showarrow=False)
            continue

        G = hvg_graph(seg, max_nodes=max_nodes)
        if G is None or G.number_of_nodes() == 0:
            continue

        pos = nx.spring_layout(G, seed=42, k=0.20, iterations=60)
        edge_x, edge_y = [], []
        for a, b in G.edges():
            edge_x += [pos[a][0], pos[b][0], None]
            edge_y += [pos[a][1], pos[b][1], None]

        deg = dict(G.degree())
        node_x = [pos[nn][0] for nn in G.nodes()]
        node_y = [pos[nn][1] for nn in G.nodes()]
        node_size = [5 + deg[nn] * 2.0 for nn in G.nodes()]
        node_text = [f"{ph}<br>n={nn}<br>grado={deg[nn]}" for nn in G.nodes()]

        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                 line=dict(width=0.45), hoverinfo="skip",
                                 showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers",
                                 marker=dict(size=node_size, opacity=0.82),
                                 text=node_text, hoverinfo="text",
                                 showlegend=False), row=r, col=c)
        fig.update_xaxes(visible=False, row=r, col=c)
        fig.update_yaxes(visible=False, row=r, col=c)

    fig.update_layout(
        height=max(650, rows * 440),
        title=f"HVG por fases · {record_name}",
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


def hvg_metrics_all_phases_figure(metrics_df):
    """
    Secuencia de métricas HVG por fases para un solo registro.
    """
    if metrics_df is None or metrics_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No hay métricas HVG")
        return fig

    hvg_cols = ["HVG_edges", "HVG_degree_mean", "HVG_degree_max", "HVG_hubs_p90",
                "HVG_clustering", "HVG_lambda", "HVG_path_length", "HVG_diameter"]
    hvg_cols = [c for c in hvg_cols if c in metrics_df.columns]
    phases = [p for p in PHASES if p in metrics_df.index]

    fig = go.Figure()
    for col in hvg_cols:
        y = [metrics_df.loc[ph, col] if ph in metrics_df.index else np.nan for ph in phases]
        fig.add_trace(go.Scatter(x=phases, y=y, mode="lines+markers", name=col, line=dict(width=3)))

    fig.update_layout(
        height=560,
        title="Secuencia de métricas HVG por fases",
        xaxis_title="Fase",
        yaxis_title="Valor",
        hovermode="x unified",
    )
    return fig




def _smooth_line_xy(y_values, smooth_points=220):
    """
    Suavizado visual seguro para líneas de tendencia.

    - 1 punto: punto único.
    - 2 puntos: línea recta inevitable.
    - 3 puntos: curva cuadrática suavizada.
    - >=4 puntos: CubicSpline natural.
    """
    y = np.asarray(y_values, dtype=float)
    x = np.arange(len(y), dtype=float)
    mask = np.isfinite(y)

    n_valid = int(np.sum(mask))
    if n_valid == 0:
        return [], []
    if n_valid == 1:
        return x[mask], y[mask]

    xs = np.linspace(x[mask].min(), x[mask].max(), smooth_points)

    try:
        if n_valid >= 4:
            cs = CubicSpline(x[mask], y[mask], bc_type="natural")
            ys = cs(xs)
        elif n_valid == 3:
            # Con tres fases/registros se puede generar una curva suave cuadrática.
            coef = np.polyfit(x[mask], y[mask], deg=2)
            ys = np.polyval(coef, xs)
        else:
            # Con dos puntos no existe suavizado real sin inventar información.
            ys = np.interp(xs, x[mask], y[mask])
    except Exception:
        ys = np.interp(xs, x[mask], y[mask])

    return xs, ys


def _add_bars_and_smooth_lines(fig, metrics_df, row, col, metrics, title, yaxis_title="Valor", secondary_y_metric=None, secondary_y_title=None):
    """
    Añade columnas verticales + líneas suavizadas superpuestas en un panel.
    """
    phases = [p for p in PHASES if p in metrics_df.index]
    if not phases:
        return

    x_base = np.arange(len(phases), dtype=float)
    present = [m for m in metrics if m in metrics_df.columns]
    n = max(1, len(present))
    bar_width = min(0.72 / n, 0.18)

    for i, m in enumerate(present):
        y = [metrics_df.loc[ph, m] if ph in metrics_df.index else np.nan for ph in phases]
        offset = (i - (n - 1) / 2) * bar_width
        x_bar = x_base + offset
        use_secondary = (secondary_y_metric is not None and m == secondary_y_metric)

        fig.add_trace(
            go.Bar(
                x=x_bar,
                y=y,
                width=bar_width,
                name=m,
                opacity=0.72,
                hovertemplate=f"{m}<br>Fase: %{{customdata}}<br>Valor: %{{y:.3f}}<extra></extra>",
                customdata=phases,
                showlegend=True,
            ),
            row=row,
            col=col,
            secondary_y=use_secondary,
        )

        xs, ys = _smooth_line_xy(y)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"{m} tendencia",
                line=dict(width=3),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
            secondary_y=use_secondary,
        )

        fig.add_trace(
            go.Scatter(
                x=x_base,
                y=y,
                mode="markers",
                name=f"{m} puntos",
                marker=dict(size=6),
                hovertemplate=f"{m}<br>Fase: %{{customdata}}<br>Valor: %{{y:.3f}}<extra></extra>",
                customdata=phases,
                showlegend=False,
            ),
            row=row,
            col=col,
            secondary_y=use_secondary,
        )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(x_base),
        ticktext=phases,
        title_text="Fase",
        row=row,
        col=col,
    )
    fig.update_yaxes(title_text=yaxis_title, row=row, col=col, secondary_y=False)
    if secondary_y_metric is not None and secondary_y_title:
        fig.update_yaxes(title_text=secondary_y_title, row=row, col=col, secondary_y=True)


def hrv_phase_summary_figure(metrics_df):
    """
    Figura tipo referencia:
    6 paneles con columnas verticales por fase y líneas de tendencia suavizadas.
    """
    if metrics_df is None or metrics_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No hay datos HRV para graficar")
        return fig

    specs = [
        [{"secondary_y": True}, {"secondary_y": False}],
        [{"secondary_y": False}, {"secondary_y": False}],
        [{"secondary_y": False}, {"secondary_y": False}],
    ]

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=specs,
        subplot_titles=[
            "1) RMSSD, SDNN, pNN50",
            "2) VLF, LF, HF, TOTAL",
            "3) SD1, SD2",
            "4) DFA α1, α2, ApEn, SampEn",
            "5) Recurrence Plot",
            "6) Multiscale Entropy (MSE 1-20)",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.11,
    )

    _add_bars_and_smooth_lines(
        fig, metrics_df, 1, 1,
        ["RMSSD", "SDNN", "pNN50"],
        "1) RMSSD, SDNN, pNN50",
        yaxis_title="ms",
        secondary_y_metric="pNN50",
        secondary_y_title="pNN50 (%)",
    )

    _add_bars_and_smooth_lines(
        fig, metrics_df, 1, 2,
        ["VLF", "LF", "HF", "TOTAL"],
        "2) VLF, LF, HF, TOTAL",
        yaxis_title="ms²",
    )

    _add_bars_and_smooth_lines(
        fig, metrics_df, 2, 1,
        ["SD1", "SD2"],
        "3) SD1, SD2",
        yaxis_title="ms",
    )

    complexity_vars = [m for m in ["DFA_alpha1", "DFA_alpha2", "D2", "ApEn", "SampEn"] if m in metrics_df.columns]
    _add_bars_and_smooth_lines(
        fig, metrics_df, 2, 2,
        complexity_vars,
        "4) DFA α1, α2, D2, ApEn, SampEn",
        yaxis_title="Valor",
    )

    recurrence_vars = [m for m in ["Lmean", "Lmax", "REC", "DET", "ShanEn"] if m in metrics_df.columns]
    _add_bars_and_smooth_lines(
        fig, metrics_df, 3, 1,
        recurrence_vars,
        "5) Recurrence Plot",
        yaxis_title="Valor",
    )

    mse_vars = [f"MSE{i}" for i in range(1, 21) if f"MSE{i}" in metrics_df.columns]
    _add_bars_and_smooth_lines(
        fig, metrics_df, 3, 2,
        mse_vars,
        "6) Multiscale Entropy (MSE 1-20)",
        yaxis_title="Valor",
    )

    fig.update_layout(
        height=1350,
        title="Resumen HRV por fases: columnas verticales + líneas suavizadas",
        barmode="group",
        hovermode="closest",
        legend_title_text="Parámetro",
        bargap=0.22,
        bargroupgap=0.02,
        margin=dict(l=60, r=40, t=100, b=70),
    )

    return fig


def hrv_phase_summary_record_panels(record_data, records_results):
    """
    Si hay varios registros: un panel visual por registro usando la misma lógica.
    Devuelve dict nombre_figura -> figura.
    """
    figs = {}
    for rec, df in records_results.items():
        if df is not None and not df.empty:
            figs[rec] = hrv_phase_summary_figure(df)
    return figs


# ============================================================
# APP
# ============================================================

st.title("VRC / HRV RRi Analyzer Pro v9.9")
st.caption("Segmentación por fases, HRV visual por fases, dominios/MSE, Poincaré, HVG e informe automático.")

with st.sidebar:
    uploaded_files = st.file_uploader("Sube uno o varios CSV/TXT con RRi", type=["csv", "txt"], accept_multiple_files=True)
    min_rr = st.number_input("Mínimo RRi por ventana", min_value=10, max_value=300, value=30, step=5)
    include_rqa = st.checkbox("Calcular RQA", value=False, help="Puede tardar en ventanas largas.")
    include_hvg = st.checkbox("Calcular HVG/grafos", value=False, help="Más lento. Actívalo cuando ya tengas las ventanas definidas.")
    artifact_level = st.selectbox(
        "Corrección de artefactos",
        ["none", "very low", "low", "medium", "strong", "very strong"],
        index=0,
        help="Aproximada tipo Kubios: mediana local + interpolación lineal.",
    )
    domain_method = st.selectbox("Cálculo dominios", ["median", "mean"], index=0)
    st.caption("Consejo: para ventanas de ~30 s usa mínimo RRi 20-30; para 5 min usa 30-110 según el caso.")

if not uploaded_files:
    st.info("Sube uno o varios registros RRi.")
    st.stop()

record_data = {}
errors = []

for uf in uploaded_files:
    try:
        rr_raw = read_rri_file(uf)
        rr, artifact_mask, artifact_info = correct_artifacts_kubios_like(rr_raw, level=artifact_level)
        name = sanitize_name(uf.name)
        base, k = name, 2

        while name in record_data:
            name = f"{base}_{k}"
            k += 1

        record_data[name] = {
            "rr": rr,
            "rr_raw": rr_raw,
            "artifact_mask": artifact_mask,
            "artifact_info": artifact_info,
            "duration": float(np.sum(rr)),
            "filename": uf.name,
        }
    except Exception as e:
        errors.append(f"{uf.name}: {e}")

if errors:
    st.error("\n".join(errors))

if not record_data:
    st.stop()

# Orden cronológico de más antiguo a más reciente usando la fecha del nombre del archivo.
record_data = sort_records_chronologically(record_data)

records = list(record_data.keys())
selected_record = st.sidebar.selectbox("Registro principal", records)
t_max = record_data[selected_record]["duration"]

# ============================================================
# Estado robusto de segmentación
# ============================================================
if "selected_record_v50" not in st.session_state or st.session_state.selected_record_v50 != selected_record:
    st.session_state.selected_record_v50 = selected_record

st.session_state.setdefault("global_windows_v50", empty_windows())
st.session_state.setdefault("record_windows_v50", {})
for rec in records:
    st.session_state.record_windows_v50.setdefault(rec, empty_windows())

st.session_state.setdefault("pending_selection_v50", None)
st.session_state.setdefault("active_phases_v50", ["Basal"])
st.session_state.setdefault("use_independent_v70", False)

with st.sidebar.expander("Segmentación", expanded=True):
    use_independent = st.checkbox("Ventanas independientes por registro", value=st.session_state.get("use_independent_v70", False), key="use_independent_checkbox_v70")
    st.session_state.use_independent_v70 = use_independent
    active_phases = st.multiselect("Fases activas para calcular", PHASES, default=st.session_state.active_phases_v50)
    st.session_state.active_phases_v50 = active_phases

    c_basal, c_rec = st.columns(2)
    with c_basal:
        if st.button("Activar basales", help="Activa Basal, Basal2, Basal3, Basal4 y Basal5"):
            st.session_state.active_phases_v50 = [p for p in PHASES if PHASE_GROUP.get(p) == "Basal"]
            st.rerun()
    with c_rec:
        if st.button("Activar recuperaciones", help="Activa R1-R6"):
            st.session_state.active_phases_v50 = [p for p in PHASES if PHASE_GROUP.get(p) == "Recuperación"]
            st.rerun()

    if st.button("Limpiar todas las ventanas"):
        st.session_state.global_windows_v50 = empty_windows()
        st.session_state.record_windows_v50 = {rec: empty_windows() for rec in records}
        st.session_state.pending_selection_v50 = None
        st.rerun()

    if st.button("Autodividir todo el registro"):
        if use_independent:
            st.session_state.record_windows_v50[selected_record] = default_windows(t_max)
        else:
            st.session_state.global_windows_v50 = default_windows(t_max)
        st.session_state.active_phases_v50 = PHASES.copy()
        st.rerun()

    if use_independent and st.button("Copiar ventanas del registro principal a todos"):
        base_w = st.session_state.record_windows_v50.get(selected_record, empty_windows())
        st.session_state.record_windows_v50 = {rec: {ph: (list(base_w[ph]) if base_w[ph] is not None else None) for ph in PHASES} for rec in records}
        st.rerun()

if artifact_level != "none":
    with st.sidebar.expander("Resumen artefactos", expanded=True):
        for rec, data in record_data.items():
            info = data.get("artifact_info", {})
            st.write(f"**{rec}**: {info.get('n_artifacts', 0)} ({info.get('percent_artifacts', 0):.2f}%)")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["1) Segmentar tipo Kubios", "2) HRV", "3) Comparar", "4) No lineales / MSE", "5) Poincaré / Grafos", "6) Dashboard", "7) Informe", "8) Exportar"])



# ============================================================
# SCRIPT LOCAL HTML -> PNG
# ============================================================



ARRANCAR_CONVERTIDOR_BAT = '@echo off\ntitle Convertidor HRV HTML a PNG\n\necho ============================================================\necho  VRC / HRV RRi Analyzer Pro - Convertidor HTML/Localhost a PNG\necho ============================================================\necho.\necho Este arrancador funciona desde cualquier carpeta porque usa %%~dp0.\necho.\n\necho Abriendo Streamlit local en el navegador...\nstart "" "http://localhost:8501/"\n\necho.\necho Esperando a que cargue la app...\ntimeout /t 5 >nul\n\necho.\necho Generando captura PNG de http://localhost:8501/ ...\npython "%~dp0capture_streamlit_localhost_png.py" "http://localhost:8501/" "%~dp0captura_streamlit.png"\n\necho.\necho Si quieres convertir los HTML exportados a PNG, ejecutando ahora:\npython "%~dp0convert_html_to_png.py"\n\necho.\necho ============================================================\necho  Proceso terminado.\necho  Captura principal:\necho  %~dp0captura_streamlit.png\necho.\necho  PNG desde HTML, si existen:\necho  %~dp0graficos\\png_from_html\necho ============================================================\necho.\npause\n'

CAPTURE_STREAMLIT_LOCALHOST_PNG_SCRIPT = r"""
# capture_streamlit_localhost_png.py
# Captura la app Streamlit o cualquier URL local como PNG.
#
# Uso:
#   1) Ejecuta tu app local:
#        streamlit run app.py
#
#   2) Instala Playwright:
#        pip install playwright
#        python -m playwright install chromium
#
#   3) Ejecuta:
#        python capture_streamlit_localhost_png.py
#
# Por defecto captura:
#        http://localhost:8501/
#
# También puedes cambiar URL y salida:
#        python capture_streamlit_localhost_png.py http://localhost:8501/ captura.png

from pathlib import Path
import sys
import asyncio
from playwright.async_api import async_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501/"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("captura_streamlit.png")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": 1920, "height": 1400},
            device_scale_factor=2
        )

        print(f"Abriendo: {URL}")
        await page.goto(URL, wait_until="networkidle")
        await page.wait_for_timeout(2000)

        # Captura página completa
        await page.screenshot(path=str(OUT), full_page=True)

        await browser.close()

    print(f"PNG guardado en: {OUT.resolve()}")

if __name__ == "__main__":
    asyncio.run(main())
"""

CONVERT_HTML_TO_PNG_SCRIPT = r"""
# convert_html_to_png.py
# Convierte todos los gráficos HTML exportados por la app a PNG.
#
# Uso local:
#   1) Instala dependencias:
#        pip install playwright
#        python -m playwright install chromium
#
#   2) Descomprime el ZIP exportado por la app.
#
#   3) Ejecuta:
#        python convert_html_to_png.py
#
# El script buscará la carpeta graficos/html y creará graficos/png_from_html.

from pathlib import Path
import asyncio
from playwright.async_api import async_playwright

BASE = Path(__file__).resolve().parent
HTML_DIR = BASE / "graficos" / "html"
OUT_DIR = BASE / "graficos" / "png_from_html"
OUT_DIR.mkdir(parents=True, exist_ok=True)

async def main():
    html_files = sorted(HTML_DIR.glob("*.html"))
    if not html_files:
        print(f"No se han encontrado HTML en: {HTML_DIR}")
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": 1800, "height": 1100},
            device_scale_factor=2
        )

        for html in html_files:
            url = html.resolve().as_uri()
            out = OUT_DIR / (html.stem + ".png")
            print(f"Convirtiendo: {html.name} -> {out.name}")
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(1200)
            await page.screenshot(path=str(out), full_page=True)

        await browser.close()

    print(f"Listo. PNG guardados en: {OUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
"""

# ============================================================
# EXPORTACIÓN DE GRÁFICOS
# ============================================================

def _safe_filename(text, max_len=90):
    """
    Nombre de archivo seguro.
    """
    text = str(text)
    text = re.sub(r"[^\w\-.]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "grafico"
    return text[:max_len]



# Paleta fija para exportación en color.
EXPORT_COLORWAY = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#00bcd4", "#ff4b4b", "#4caf50", "#ffc107", "#9c27b0",
    "#03a9f4", "#ff9800", "#8bc34a", "#f44336", "#673ab7",
]

def _export_color_for(i):
    return EXPORT_COLORWAY[int(i) % len(EXPORT_COLORWAY)]


def _trace_has_color(trace, attr_path):
    """
    Comprueba si un trace tiene color explícito.
    attr_path ejemplos:
    - ("marker", "color")
    - ("line", "color")
    """
    try:
        obj = trace
        for attr in attr_path:
            obj = getattr(obj, attr)
        if obj is None:
            return False
        if isinstance(obj, (list, tuple, np.ndarray)):
            return len(obj) > 0
        return str(obj) != ""
    except Exception:
        return False


def _prepare_plotly_fig_for_color_export(fig):
    """
    Prepara una copia de la figura para exportar con colores fijos.

    Motivo:
    En Streamlit los gráficos pueden verse coloreados por el tema del navegador,
    pero al exportar con Kaleido/HTML fuera de Streamlit algunos traces sin color explícito
    pueden salir grises. Aquí se fija:
    - plantilla oscura,
    - colorway,
    - fondo oscuro,
    - color explícito en barras, líneas, marcadores y redes.
    """
    try:
        f = copy.deepcopy(fig)
    except Exception:
        f = fig

    try:
        f.update_layout(
            template="plotly_dark",
            colorway=EXPORT_COLORWAY,
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="#FAFAFA"),
            legend=dict(
                bgcolor="rgba(14,17,23,0.75)",
                bordercolor="rgba(255,255,255,0.15)",
                borderwidth=1,
                font=dict(color="#FAFAFA"),
            ),
        )

        for ax in list(f.layout):
            if str(ax).startswith("xaxis") or str(ax).startswith("yaxis"):
                try:
                    f.layout[ax].update(
                        gridcolor="rgba(255,255,255,0.12)",
                        zerolinecolor="rgba(255,255,255,0.20)",
                        linecolor="rgba(255,255,255,0.25)",
                        tickfont=dict(color="#FAFAFA"),
                        titlefont=dict(color="#FAFAFA"),
                    )
                except Exception:
                    pass

        for i, tr in enumerate(f.data):
            col = _export_color_for(i)

            # Barras
            if getattr(tr, "type", "") == "bar":
                if not _trace_has_color(tr, ("marker", "color")):
                    tr.marker.color = col
                try:
                    tr.marker.line.color = "rgba(255,255,255,0.25)"
                    tr.marker.line.width = 0.5
                except Exception:
                    pass

            # Líneas y puntos
            if getattr(tr, "type", "") == "scatter":
                mode = str(getattr(tr, "mode", "") or "")

                if "lines" in mode:
                    if not _trace_has_color(tr, ("line", "color")):
                        tr.line.color = col
                    if not getattr(tr.line, "width", None):
                        tr.line.width = 2.5

                if "markers" in mode:
                    if not _trace_has_color(tr, ("marker", "color")):
                        tr.marker.color = col
                    if not getattr(tr.marker, "size", None):
                        tr.marker.size = 7

                # Si son aristas de grafos sin color, dar gris azulado visible, no negro.
                if mode == "lines" and (getattr(tr, "showlegend", None) is False):
                    name = str(getattr(tr, "name", "") or "").lower()
                    if ("edge" in name) or ("arista" in name) or len(getattr(tr, "x", []) or []) > 100:
                        if not _trace_has_color(tr, ("line", "color")):
                            tr.line.color = "rgba(120,180,255,0.35)"
                            tr.line.width = 0.8

    except Exception:
        pass

    return f


def _write_plotly_html(fig, out_path, title=None):
    """
    Guarda una figura Plotly como HTML interactivo manteniendo colores fijos.
    """
    try:
        fig_export = _prepare_plotly_fig_for_color_export(fig)
        fig_export.write_html(
            str(out_path),
            include_plotlyjs="cdn",
            full_html=True,
            config={
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": _safe_filename(title or out_path.stem),
                    "height": 1000,
                    "width": 1600,
                    "scale": 2,
                },
                "displaylogo": False,
            },
        )
        return True
    except Exception:
        return False


def _try_write_plotly_png(fig, out_path):
    """
    Intenta guardar PNG en color si Kaleido está disponible.
    """
    try:
        fig_export = _prepare_plotly_fig_for_color_export(fig)
        fig_export.write_image(
            str(out_path),
            width=1800,
            height=1100,
            scale=2,
            format="png",
        )
        return True
    except Exception:
        return False


def build_all_export_figures(
    record_data,
    records_results,
    long_df,
    records,
    selected_record,
    global_windows,
    record_windows,
    active_phases,
    use_independent,
    domain_method,
    include_hvg,
    dashboard_params=None,
    dashboard_phases=None,
):
    """
    Construye todos los gráficos principales de la app para exportarlos.

    Exporta HTML interactivo:
    - resumen HRV por registro,
    - dominios por registro,
    - MSE por registro,
    - comparativa MSE entre registros,
    - dashboard comparativo,
    - Poincaré por fases y por fase,
    - HVG por fases, comparativo e individual si está activado.
    """
    figures = []

    def add(name, fig):
        try:
            if fig is not None:
                figures.append((name, fig))
        except Exception:
            pass

    available_phases = []
    try:
        available_phases = [p for p in PHASES if p in long_df["Fase"].unique()]
    except Exception:
        available_phases = active_phases or ["Basal"]

    if not available_phases:
        available_phases = active_phases or ["Basal"]

    # 1) HRV, dominios y MSE por registro
    for rec in records:
        dfrec = records_results.get(rec, pd.DataFrame())
        if dfrec is None or dfrec.empty:
            continue

        add(f"01_HRV_resumen_{rec}", hrv_phase_summary_figure(dfrec))
        add(f"02_Dominios_{rec}", domains_figure(dfrec, method=domain_method, title=f"Dominios · {rec}"))
        add(f"03_MSE_1_20_{rec}", mse_figure(dfrec, title=f"MSE 1-20 · {rec}"))

        # Poincaré: todas las fases del registro
        try:
            add(
                f"04_Poincare_panel_fases_{rec}",
                poincare_all_phases_panel_figure(
                    record_data,
                    global_windows,
                    record_windows,
                    rec,
                    use_independent,
                ),
            )
        except Exception:
            pass

        # HVG: todas las fases del registro
        if include_hvg:
            try:
                add(
                    f"05_HVG_panel_fases_{rec}",
                    hvg_all_phases_panel_figure(
                        record_data,
                        global_windows,
                        record_windows,
                        rec,
                        use_independent,
                        max_nodes=120,
                    ),
                )
            except Exception:
                pass

            try:
                add(
                    f"06_HVG_metricas_fases_{rec}",
                    hvg_metrics_all_phases_figure(dfrec),
                )
            except Exception:
                pass

    # 2) Comparativas entre registros
    if long_df is not None and not long_df.empty:
        # Dashboard general
        try:
            numeric_vars = [
                c for c in long_df.columns
                if c not in ["Registro", "Fase"] and pd.api.types.is_numeric_dtype(long_df[c])
            ]
            default_params = [p for p in (dashboard_params or DEFAULT_MULTI) if p in numeric_vars]
            if not default_params:
                default_params = numeric_vars[:8]
            phases_for_dash = dashboard_phases or available_phases
            if default_params:
                add(
                    "10_Dashboard_comparativo_barras_linea_suavizada",
                    dashboard_bar_smooth(long_df, phases_for_dash, default_params),
                )
        except Exception:
            pass

        # Comparativas individuales de parámetros clave
        try:
            key_params = [
                "RMSSD", "SDNN", "SD1", "SD2", "LF", "HF", "TOTAL",
                "DFA_alpha1", "DFA_alpha2", "ApEn", "SampEn", "REC", "DET",
                "Lmean", "Lmax", "ShanEn"
            ]
            for param in key_params:
                if param in long_df.columns and pd.api.types.is_numeric_dtype(long_df[param]):
                    pivot = long_df.pivot_table(index="Fase", columns="Registro", values=param, aggfunc="first")
                    if pivot is not None and not pivot.empty:
                        add(f"11_Comparativa_{param}", comparison_bar_line(pivot, param))
        except Exception:
            pass

        # Comparativa MSE
        try:
            add(
                "12_Comparativa_MSE_1_20",
                mse_compare_figure(long_df, available_phases, scales=list(range(1, 21))),
            )
        except Exception:
            pass

        # RRi superpuesto por fase
        for ph in available_phases:
            try:
                add(
                    f"13_RRi_superpuesto_{ph}",
                    phase_rr_overlay(record_data, global_windows, record_windows, ph, use_independent),
                )
            except Exception:
                pass

        # Poincaré por fase: paneles separados y superpuestos
        for ph in available_phases:
            try:
                add(
                    f"14_Poincare_panel_{ph}",
                    poincare_panel_figure(record_data, global_windows, record_windows, ph, use_independent),
                )
            except Exception:
                pass
            try:
                add(
                    f"15_Poincare_superpuesto_{ph}",
                    poincare_figure(record_data, global_windows, record_windows, ph, use_independent),
                )
            except Exception:
                pass

        # HVG por fase
        if include_hvg:
            for ph in available_phases:
                try:
                    add(
                        f"16_HVG_comparativo_{ph}",
                        hvg_network_compare_figure(
                            record_data,
                            global_windows,
                            record_windows,
                            ph,
                            use_independent,
                            max_nodes=120,
                        ),
                    )
                except Exception:
                    pass

                try:
                    hvg_cols_export = [
                        "HVG_graph_score_scale_free",
                        "HVG_graph_score_small_world",
                        "HVG_graph_score_chain",
                        "HVG_compactness_index",
                        "HVG_nodes",
                        "HVG_edges",
                        "HVG_degree_mean",
                        "HVG_degree_max",
                        "HVG_hubs_p90",
                        "HVG_clustering",
                        "HVG_lambda",
                        "HVG_path_length",
                        "HVG_diameter",
                    ]
                    hvg_df = long_df[long_df["Fase"] == ph]
                    for hvg_param in hvg_cols_export:
                        if hvg_param in hvg_df.columns and pd.api.types.is_numeric_dtype(hvg_df[hvg_param]):
                            pivot_hvg = hvg_df.pivot_table(index="Fase", columns="Registro", values=hvg_param, aggfunc="first")
                            if pivot_hvg is not None and not pivot_hvg.empty:
                                add(f"17_HVG_metrica_{hvg_param}_{ph}", comparison_bar_line(pivot_hvg, hvg_param))
                except Exception:
                    pass

    return figures


def write_all_graph_exports(figures, outdir, formats=("html",)):
    """
    Guarda todos los gráficos en los formatos indicados.

    Formatos:
    - html: siempre recomendado; no depende de motores externos.
    - png: usa Plotly + Kaleido. Si Kaleido no está disponible, no rompe la app.
    - svg: usa Plotly + Kaleido. Útil para publicaciones.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    formats = set(formats or ["html"])

    html_dir = outdir / "html"
    png_dir = outdir / "png"
    svg_dir = outdir / "svg"

    if "html" in formats:
        html_dir.mkdir(parents=True, exist_ok=True)
    if "png" in formats:
        png_dir.mkdir(parents=True, exist_ok=True)
    if "svg" in formats:
        svg_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    used_names = set()
    png_errors = []

    for i, (name, fig) in enumerate(figures, start=1):
        base = _safe_filename(f"{i:03d}_{name}")
        while base in used_names:
            base = _safe_filename(base + "_copy")
        used_names.add(base)

        html_name = ""
        png_name = ""
        svg_name = ""
        ok_html = False
        ok_png = False
        ok_svg = False

        if "html" in formats:
            html_path = html_dir / f"{base}.html"
            ok_html = _write_plotly_html(fig, html_path, title=name)
            if ok_html:
                html_name = f"html/{html_path.name}"

        if "png" in formats:
            png_path = png_dir / f"{base}.png"
            ok_png = _try_write_plotly_png(fig, png_path)
            if ok_png:
                png_name = f"png/{png_path.name}"
            else:
                png_errors.append(base)

        if "svg" in formats:
            try:
                svg_path = svg_dir / f"{base}.svg"
                _prepare_plotly_fig_for_color_export(fig).write_image(str(svg_path), width=1800, height=1100, scale=1, format="svg")
                ok_svg = True
                svg_name = f"svg/{svg_path.name}"
            except Exception:
                ok_svg = False

        index_rows.append({
            "N": i,
            "Grafico": name,
            "HTML": html_name,
            "PNG": png_name,
            "SVG": svg_name,
            "Exportado_HTML": ok_html,
            "Exportado_PNG": ok_png,
            "Exportado_SVG": ok_svg,
        })

    index_df = pd.DataFrame(index_rows)
    index_path = outdir / "indice_graficos_exportados.csv"
    index_df.to_csv(index_path, index=False)

    if png_errors:
        (outdir / "AVISO_PNG.txt").write_text(
            "Algunos PNG/SVG no se han podido generar automáticamente.\n\n"
            "Motivo habitual: falta Kaleido o Chrome/Chromium en Streamlit Cloud.\n\n"
            "Soluciones:\n"
            "1) Añadir kaleido a requirements.txt.\n"
            "2) Mantener exportación HTML, que siempre funciona.\n"
            "3) Usar el script convert_html_to_png.py incluido en este ZIP en tu ordenador local.\n\n"
            f"Gráficos no convertidos: {len(png_errors)}\n",
            encoding="utf-8"
        )

    return index_df


# central calculation
records_results, records_segments, records_valid = {}, {}, {}

global_windows_safe = st.session_state.get("global_windows_v50", empty_windows())
record_windows_safe = st.session_state.get("record_windows_v50", {})
for rec in records:
    record_windows_safe.setdefault(rec, empty_windows())
active_phases = st.session_state.get("active_phases_v50", ["Basal"])
use_independent = st.session_state.get("use_independent_v70", False)

for rec, data in record_data.items():
    w = get_record_windows(global_windows_safe, record_windows_safe, rec, use_independent)
    df, segs, valid = calculate_record(data["rr"], w, active_phases, min_rr, include_rqa, include_hvg=include_hvg)
    records_results[rec], records_segments[rec], records_valid[rec] = df, segs, valid

metrics_df = records_results[selected_record]
long_df = build_long(records_results)

with tab1:
    st.subheader("Segmentación tipo Kubios")
    st.write(
        "1) Encuadra una región con el ratón. "
        "2) Pulsa **Guardar selección**. "
        "3) Pulsa **Asignar a Basal/E1/E2...**. "
        "Sólo se calcularán las fases activas."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        view_mode = st.radio("Vista", ["Registro principal", "Todos superpuestos"], index=1)
    with c2:
        st.info("Para comparar dos registros del mismo paciente, usa 'Todos superpuestos' y asigna las ventanas que quieras comparar.")

    fig = rr_plot(
        record_data,
        st.session_state.global_windows_v50,
        st.session_state.record_windows_v50,
        view_mode,
        selected_record,
        use_independent,
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode=("box", "lasso"),
        key="rr_select_v50",
    )

    if event and getattr(event, "selection", None):
        pts = event.selection.get("points", [])
        xs = [p.get("x") for p in pts if "x" in p]

        if xs:
            s_sel, e_sel = min(xs) * 60, max(xs) * 60
            st.success(f"Selección detectada: {sec_to_hms(s_sel)} - {sec_to_hms(e_sel)}")

            if st.button("Guardar selección"):
                st.session_state.pending_selection_v50 = [s_sel, e_sel]
                st.rerun()

    if st.session_state.pending_selection_v50 is not None:
        s_sel, e_sel = st.session_state.pending_selection_v50
        st.success(f"Selección guardada: {sec_to_hms(s_sel)} - {sec_to_hms(e_sel)}")

        st.markdown("### Asignar selección guardada a fase")
        phase_cols = st.columns(10)

        for idx, ph in enumerate(PHASES):
            with phase_cols[idx % 10]:
                if st.button(ph, key=f"assign_{ph}_v50"):
                    if use_independent:
                        st.session_state.record_windows_v50[selected_record][ph] = [s_sel, e_sel]
                    else:
                        st.session_state.global_windows_v50[ph] = [s_sel, e_sel]

                    if ph not in st.session_state.active_phases_v50:
                        st.session_state.active_phases_v50.append(ph)

                    st.session_state.pending_selection_v50 = None
                    st.rerun()

        if st.button("Borrar selección guardada"):
            st.session_state.pending_selection_v50 = None
            st.rerun()

    st.markdown("### Ventanas definidas")
    win_df = windows_table(
        st.session_state.global_windows_v50,
        st.session_state.record_windows_v50,
        records,
        record_data,
        records_segments,
        records_valid,
        use_independent,
    )
    st.dataframe(win_df, use_container_width=True)

    st.markdown("### Edición manual opcional")
    manual_phase = st.selectbox("Fase a editar manualmente", PHASES)
    current_w = get_record_windows(st.session_state.global_windows_v50, st.session_state.record_windows_v50, selected_record, use_independent).get(manual_phase)

    if current_w is None:
        ini_default, fin_default = "00:00:00", "00:05:00"
    else:
        ini_default, fin_default = sec_to_hms(current_w[0]), sec_to_hms(current_w[1])

    c_ini, c_fin, c_apply, c_clear = st.columns([1, 1, 1, 1])
    with c_ini:
        ini_txt = st.text_input("Inicio", ini_default)
    with c_fin:
        fin_txt = st.text_input("Fin", fin_default)
    with c_apply:
        st.write("")
        st.write("")
        if st.button("Aplicar manual"):
            try:
                s, e = hms_to_sec(ini_txt), hms_to_sec(fin_txt)
                if e <= s:
                    st.warning("El final debe ser mayor que el inicio.")
                else:
                    if use_independent:
                        st.session_state.record_windows_v50[selected_record][manual_phase] = [s, e]
                    else:
                        st.session_state.global_windows_v50[manual_phase] = [s, e]
                    if manual_phase not in st.session_state.active_phases_v50:
                        st.session_state.active_phases_v50.append(manual_phase)
                    st.rerun()
            except Exception:
                st.warning("Formato no válido. Usa HH:MM:SS.")
    with c_clear:
        st.write("")
        st.write("")
        if st.button("Borrar fase"):
            if use_independent:
                st.session_state.record_windows_v50[selected_record][manual_phase] = None
            else:
                st.session_state.global_windows_v50[manual_phase] = None
            if manual_phase in st.session_state.active_phases_v50:
                st.session_state.active_phases_v50.remove(manual_phase)
            st.rerun()

with tab2:
    st.subheader(f"HRV: {selected_record}")

    if metrics_df.empty:
        st.info("No hay ventanas válidas para el registro principal. Define ventanas, activa fases o baja el mínimo RRi.")
    else:
        st.markdown("### Resumen visual por fases")
        st.caption("Columnas verticales = valores por fase. Líneas = tendencia suavizada superpuesta.")
        st.plotly_chart(
            hrv_phase_summary_figure(metrics_df),
            use_container_width=True,
            key=f"hrv_summary_{selected_record}"
        )

        for group, cols in PARAM_GROUPS.items():
            present = [c for c in cols if c in metrics_df.columns]
            if present:
                st.markdown(f"### {group}")
                st.dataframe(metrics_df[present], use_container_width=True)

with tab3:
    st.subheader("Comparar registros")

    if len(records) < 2:
        st.info("Sube dos o más registros.")
    elif long_df.empty:
        st.info("No hay datos comparables. Define ventanas, activa fases o baja el mínimo RRi.")
    else:
        valid_summary = pd.DataFrame(records_valid).T.reindex(columns=PHASES)
        st.markdown("### Ventanas válidas")
        st.dataframe(valid_summary, use_container_width=True)

        available_phases = [p for p in PHASES if p in long_df["Fase"].unique()]
        selected_phases = st.multiselect("Fases a comparar", PHASES, default=available_phases)
        numeric_vars = [c for c in long_df.columns if c not in ["Registro", "Fase"] and pd.api.types.is_numeric_dtype(long_df[c])]

        default_var = "RMSSD" if "RMSSD" in numeric_vars else numeric_vars[0]
        variable = st.selectbox("Variable principal", numeric_vars, index=numeric_vars.index(default_var))
        df_sel = long_df[long_df["Fase"].isin(selected_phases)] if selected_phases else long_df
        pivot = df_sel.pivot_table(index="Fase", columns="Registro", values=variable, aggfunc="first").reindex(selected_phases)

        st.markdown(f"### {variable}: barras agrupadas + línea de tendencia")
        st.dataframe(pivot, use_container_width=True)
        st.plotly_chart(comparison_bar_line(pivot, variable), use_container_width=True, key=f"compare_main_{variable}_{len(selected_phases)}")

        st.markdown("### Panel de varios parámetros: barras + línea suavizada")
        param_defaults = [p for p in DEFAULT_MULTI if p in numeric_vars]
        params = st.multiselect("Parámetros", numeric_vars, default=param_defaults)
        if params:
            st.plotly_chart(dashboard_bar_smooth(long_df, selected_phases or available_phases, params), use_container_width=True, key="compare_dashboard_params_smooth")

        ph_overlay = st.selectbox("RRi superpuesto por fase", selected_phases or available_phases)
        st.plotly_chart(
            phase_rr_overlay(record_data, st.session_state.global_windows_v50, st.session_state.record_windows_v50, ph_overlay, use_independent),
            use_container_width=True,
            key=f"phase_overlay_{ph_overlay}",
        )

        st.markdown("### Tabla completa filtrada")
        st.dataframe(df_sel, use_container_width=True)



with tab4:
    st.subheader("Parámetros no lineales: dominios y MSE 1-20")

    if metrics_df.empty:
        st.info("No hay ventanas válidas para mostrar dominios o MSE.")
    else:
        st.markdown("### Dominios Amplitud / Vagal / Complejidad / Recurrencia")
        st.caption("Normalizado a Basal = 100%. Amplitud: SDNN, SD2, Total Power. Vagal: RMSSD, SD1, HF, pNN50. Complejidad: DFA α1, DFA α2, ApEn, SampEn. Recurrencia: REC, DET, Lmean, Lmax, ShanEn.")
        st.plotly_chart(
            domains_figure(metrics_df, method=domain_method, title=f"Dominios · {selected_record}"),
            use_container_width=True,
            key="domains_principal"
        )
        st.dataframe(domain_values(metrics_df, method=domain_method), use_container_width=True)

        st.markdown("### MSE 1-20 del registro principal")
        st.plotly_chart(
            mse_figure(metrics_df, title=f"MSE 1-20 · {selected_record}"),
            use_container_width=True,
            key="mse_principal"
        )

    if not long_df.empty and len(records) >= 2:
        st.markdown("### Comparativa MSE 1-20 entre registros")
        available_phases_mse = [p for p in PHASES if p in long_df["Fase"].unique()]
        phases_mse = st.multiselect("Fases para comparar MSE", PHASES, default=available_phases_mse, key="mse_compare_phases")
        scale_range = st.slider("Escalas MSE", 1, 20, (1, 20), key="mse_scale_range")
        scales = list(range(scale_range[0], scale_range[1] + 1))
        st.plotly_chart(
            mse_compare_figure(long_df, phases_mse or available_phases_mse, scales=scales),
            use_container_width=True,
            key="mse_compare"
        )


with tab5:
    st.subheader("Poincaré y grafos comparativos")

    if len(records) < 1:
        st.info("Sube al menos un registro.")
    else:
        available_phases_pg = [p for p in PHASES if p in active_phases]
        if not available_phases_pg:
            available_phases_pg = [p for p in PHASES if any(records_valid[rec].get(p, False) for rec in records)]

        if not available_phases_pg:
            st.info("No hay fases válidas. Define ventanas y activa fases.")
        else:
            phase_pg = st.selectbox("Fase para Poincaré / grafo", available_phases_pg, key="phase_pg_v51")
            modo_fases_pg = st.radio(
                "Qué quieres mostrar",
                ["Una fase seleccionada", "Todas las fases del registro principal"],
                horizontal=True,
                key="modo_fases_pg_v71"
            )

            st.markdown("### Poincaré")
            if modo_fases_pg == "Todas las fases del registro principal":
                st.plotly_chart(
                    poincare_all_phases_panel_figure(
                        record_data,
                        st.session_state.global_windows_v50,
                        st.session_state.record_windows_v50,
                        selected_record,
                        use_independent,
                    ),
                    use_container_width=True,
                    key=f"poincare_all_phases_{selected_record}"
                )
            else:
                modo_poincare = st.radio(
                    "Modo de visualización Poincaré",
                    ["Paneles separados", "Superpuestos"],
                    horizontal=True,
                    key="modo_poincare_v63"
                )

                if modo_poincare == "Paneles separados":
                    st.plotly_chart(
                        poincare_panel_figure(
                            record_data,
                            st.session_state.global_windows_v50,
                            st.session_state.record_windows_v50,
                            phase_pg,
                            use_independent,
                        ),
                        use_container_width=True,
                        key=f"poincare_panel_{phase_pg}"
                    )
                else:
                    st.plotly_chart(
                        poincare_figure(
                            record_data,
                            st.session_state.global_windows_v50,
                            st.session_state.record_windows_v50,
                            phase_pg,
                            use_independent,
                        ),
                        use_container_width=True,
                        key=f"poincare_overlay_{phase_pg}"
                    )

            st.markdown("### Métricas HVG / grafos")
            try:
                selected_metrics_df = records_results.get(selected_record, pd.DataFrame())
                if selected_metrics_df is not None and not selected_metrics_df.empty:
                    st.markdown("#### Resumen topológico HVG")
                    first_valid_hvg = None
                    for _fase, _row in selected_metrics_df.iterrows():
                        if "HVG_nodes" in selected_metrics_df.columns and pd.notna(_row.get("HVG_nodes", np.nan)):
                            first_valid_hvg = _row.to_dict()
                            break
                    if first_valid_hvg is not None:
                        st.dataframe(hvg_summary_card(first_valid_hvg), use_container_width=True)
                        st.info(str(first_valid_hvg.get("HVG_topology_interpretation", "")))
                st.markdown("#### Definiciones y rangos orientativos")
                st.dataframe(hvg_reference_ranges(), use_container_width=True)
            except Exception:
                pass
            if not include_hvg:
                st.warning("Activa 'Calcular HVG/grafos' en la barra lateral para calcular las métricas de grafos.")
            else:
                hvg_cols = [
        "HVG_graph_type", "HVG_graph_score_scale_free", "HVG_graph_score_small_world", "HVG_graph_score_chain",
        "HVG_nodes", "HVG_edges", "HVG_degree_mean", "HVG_degree_max", "HVG_hubs_p90",
        "HVG_clustering", "HVG_lambda", "HVG_path_length", "HVG_diameter", "HVG_graph_interpretation"
    ]
                if "Fase" in long_df.columns:
                    hvg_df = long_df[long_df["Fase"] == phase_pg][["Registro", "Fase"] + [c for c in hvg_cols if c in long_df.columns]]
                else:
                    hvg_df = pd.DataFrame(columns=["Registro", "Fase"] + [c for c in hvg_cols if c in long_df.columns])
                if hvg_df.empty:
                    st.info("No hay métricas HVG disponibles para esta fase. Revisa que la fase esté activa y tenga suficientes RRi.")
                else:
                    st.dataframe(hvg_df, use_container_width=True)

                hvg_numeric = [c for c in hvg_cols if c in hvg_df.columns and pd.api.types.is_numeric_dtype(hvg_df[c])]
                if hvg_numeric and not hvg_df.empty:
                    hvg_var = st.selectbox("Métrica de grafo a comparar", hvg_numeric)
                    pivot_hvg = hvg_df.pivot_table(index="Fase", columns="Registro", values=hvg_var, aggfunc="first")
                    st.plotly_chart(comparison_bar_line(pivot_hvg, hvg_var), use_container_width=True, key=f"hvg_compare_{hvg_var}_{phase_pg}")

                if modo_fases_pg == "Todas las fases del registro principal":
                    st.markdown("### Grafos HVG por fases")
                    st.caption("Se muestran todas las fases del registro principal en paneles.")
                    st.plotly_chart(
                        hvg_all_phases_panel_figure(
                            record_data,
                            st.session_state.global_windows_v50,
                            st.session_state.record_windows_v50,
                            selected_record,
                            use_independent,
                            max_nodes=120
                        ),
                        use_container_width=True,
                        key=f"hvg_all_phases_{selected_record}"
                    )
                    st.plotly_chart(
                        hvg_metrics_all_phases_figure(records_results.get(selected_record, pd.DataFrame())),
                        use_container_width=True,
                        key=f"hvg_metrics_all_phases_{selected_record}"
                    )
                else:
                    st.markdown("### Grafos HVG comparativos")
                    st.caption("Se muestran los grafos de los registros lado a lado para la misma fase.")
                    st.plotly_chart(
                        hvg_network_compare_figure(
                            record_data,
                            st.session_state.global_windows_v50,
                            st.session_state.record_windows_v50,
                            phase_pg,
                            use_independent,
                            max_nodes=120
                        ),
                        use_container_width=True,
                        key=f"hvg_network_compare_{phase_pg}"
                    )

                st.markdown("### Grafo HVG individual")
                rec_graph = st.selectbox("Registro para visualizar individual", records, key="rec_graph_v53")
                windows_graph = get_record_windows(
                    st.session_state.global_windows_v50,
                    st.session_state.record_windows_v50,
                    rec_graph,
                    use_independent
                )
                w_graph = windows_graph.get(phase_pg)
                if w_graph is not None:
                    seg_graph = cut_segment(record_data[rec_graph]["rr"], w_graph[0], w_graph[1])
                    if len(seg_graph) >= min_rr:
                        st.plotly_chart(
                            hvg_network_figure(seg_graph, title=f"HVG {rec_graph} · {phase_pg}", max_nodes=140),
                            use_container_width=True,
                            key=f"hvg_network_individual_{rec_graph}_{phase_pg}"
                        )
                    else:
                        st.info("La fase seleccionada tiene pocos RRi para visualizar el grafo.")


with tab6:
    st.subheader("Dashboard visual: barras + línea suavizada")

    if long_df.empty:
        st.info("No hay datos.")
    else:
        st.markdown("### Resumen HRV del registro principal")
        st.plotly_chart(
            hrv_phase_summary_figure(metrics_df),
            use_container_width=True,
            key=f"dashboard_hrv_summary_{selected_record}"
        )

        available_phases = [p for p in PHASES if p in long_df["Fase"].unique()]
        numeric_vars = [c for c in long_df.columns if c not in ["Registro", "Fase"] and pd.api.types.is_numeric_dtype(long_df[c])]
        phases_dash = st.multiselect("Fases", PHASES, default=available_phases, key="dash_phases")
        params_dash = st.multiselect("Parámetros", numeric_vars, default=[p for p in DEFAULT_MULTI if p in numeric_vars], key="dash_params")
        if params_dash:
            st.plotly_chart(dashboard_bar_smooth(long_df, phases_dash or available_phases, params_dash), use_container_width=True, key="dashboard_tab_smooth")


with tab7:
    st.subheader("Informe automático HRV + grafos")
    report_md = generate_auto_report(
        record_data,
        records_results,
        st.session_state.global_windows_v50,
        st.session_state.record_windows_v50,
        active_phases,
        use_independent,
        long_df,
    )
    st.markdown(report_md)
    report_html = markdown_to_simple_html(report_md)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Descargar informe Markdown", report_md.encode("utf-8"), file_name="informe_hrv_grafos.md", mime="text/markdown")
    with c2:
        st.download_button("Descargar informe HTML", report_html.encode("utf-8"), file_name="informe_hrv_grafos.html", mime="text/html")



with tab8:
    st.subheader("Exportar")

    if long_df.empty:
        st.info("No hay datos para exportar.")
    else:
        valid_summary = pd.DataFrame(records_valid).T.reindex(columns=PHASES)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            xlsx = tmpdir / "resultados_hrv_comparativa.xlsx"
            csv = tmpdir / "resultados_hrv_comparativa.csv"
            zipf = tmpdir / "resultados_hrv_comparativa.zip"

            long_df.to_csv(csv, index=False)

            with pd.ExcelWriter(xlsx) as writer:
                long_df.to_excel(writer, sheet_name="metricas", index=False)
                valid_summary.to_excel(writer, sheet_name="ventanas_validas")

                rows_w = []
                for rec in records:
                    w = get_record_windows(st.session_state.global_windows_v50, st.session_state.record_windows_v50, rec, use_independent)
                    for ph in PHASES:
                        ww = w.get(ph)
                        rows_w.append({
                            "Registro": rec,
                            "Fase": ph,
                            "Inicio": sec_to_hms(ww[0]) if ww else "",
                            "Fin": sec_to_hms(ww[1]) if ww else "",
                            "Duracion_min": (ww[1] - ww[0]) / 60 if ww else np.nan,
                            "Activa": ph in active_phases,
                        })
                pd.DataFrame(rows_w).to_excel(writer, sheet_name="ventanas", index=False)

                artifact_rows = []
                for rec, data in record_data.items():
                    info = data.get("artifact_info", {})
                    artifact_rows.append({
                        "Registro": rec,
                        "Nivel_correccion": info.get("level", "none"),
                        "Artefactos_n": info.get("n_artifacts", 0),
                        "Artefactos_pct": info.get("percent_artifacts", 0.0),
                    })
                # Dominios por registro
                dom_rows = []
                for rec, dfrec in records_results.items():
                    dom = domain_values(dfrec, method=domain_method)
                    if not dom.empty:
                        tmp_dom = dom.copy()
                        tmp_dom.insert(0, "Registro", rec)
                        tmp_dom.insert(1, "Fase", tmp_dom.index)
                        dom_rows.append(tmp_dom.reset_index(drop=True))
                if dom_rows:
                    pd.concat(dom_rows, ignore_index=True).to_excel(writer, sheet_name="dominios", index=False)

                # MSE formato largo
                mse_cols_export = [c for c in MSE_COLUMNS if c in long_df.columns]
                if mse_cols_export:
                    long_df[["Registro", "Fase"] + mse_cols_export].to_excel(writer, sheet_name="MSE_1_20", index=False)

                pd.DataFrame(artifact_rows).to_excel(writer, sheet_name="artefactos", index=False)
                report_preview = generate_auto_report(
                    record_data,
                    records_results,
                    st.session_state.global_windows_v50,
                    st.session_state.record_windows_v50,
                    active_phases,
                    use_independent,
                    long_df,
                )
                pd.DataFrame({"Informe": report_preview.splitlines()}).to_excel(writer, sheet_name="informe", index=False)

            report_md = generate_auto_report(
                record_data,
                records_results,
                st.session_state.global_windows_v50,
                st.session_state.record_windows_v50,
                active_phases,
                use_independent,
                long_df,
            )
            report_html = markdown_to_simple_html(report_md)
            p_report_md = tmpdir / "informe_hrv_grafos.md"
            p_report_html = tmpdir / "informe_hrv_grafos.html"
            p_report_md.write_text(report_md, encoding="utf-8")
            p_report_html.write_text(report_html, encoding="utf-8")

            st.markdown("### Gráficos")
            export_formats = st.multiselect(
                "Formatos de gráficos a incluir en el ZIP (color fijo)",
                ["PNG", "SVG", "HTML interactivo"],
                default=["PNG", "HTML interactivo"],
                help=(
                    "PNG/SVG requieren Kaleido en el servidor. "
                    "HTML interactivo siempre funciona y además puede convertirse localmente a PNG con el script incluido."
                ),
                key="export_graph_formats_v88",
            )

            formats_internal = []
            if "HTML interactivo" in export_formats:
                formats_internal.append("html")
            if "PNG" in export_formats:
                formats_internal.append("png")
            if "SVG" in export_formats:
                formats_internal.append("svg")
            if not formats_internal:
                formats_internal = ["html"]

            graphs_dir = tmpdir / "graficos"
            figures_to_export = build_all_export_figures(
                record_data=record_data,
                records_results=records_results,
                long_df=long_df,
                records=records,
                selected_record=selected_record,
                global_windows=st.session_state.global_windows_v50,
                record_windows=st.session_state.record_windows_v50,
                active_phases=active_phases,
                use_independent=use_independent,
                domain_method=domain_method,
                include_hvg=include_hvg,
                dashboard_params=st.session_state.get("dash_params", None),
                dashboard_phases=st.session_state.get("dash_phases", None),
            )

            index_graphs = write_all_graph_exports(figures_to_export, graphs_dir, formats=formats_internal)

            # Script local para convertir HTML exportados a PNG si Kaleido no funciona en Streamlit Cloud.
            converter_script = tmpdir / "convert_html_to_png.py"
            try:
                converter_script.write_text(CONVERT_HTML_TO_PNG_SCRIPT, encoding="utf-8")
            except Exception:
                converter_script.write_text("# Script de conversión no disponible en esta versión.\n", encoding="utf-8")

            localhost_capture_script = tmpdir / "capture_streamlit_localhost_png.py"
            try:
                localhost_capture_script.write_text(CAPTURE_STREAMLIT_LOCALHOST_PNG_SCRIPT, encoding="utf-8")
            except Exception:
                localhost_capture_script.write_text("# Script de captura localhost no disponible en esta versión.\n", encoding="utf-8")

            arrancador_bat = tmpdir / "Arrancar_Convertidor.bat"
            arrancador_bat.write_text(globals().get("ARRANCAR_CONVERTIDOR_BAT", "@echo off\nstart \"\" \"http://localhost:8501/\"\npython \"%~dp0capture_streamlit_localhost_png.py\" \"http://localhost:8501/\" \"%~dp0captura_streamlit.png\"\npause\n"), encoding="utf-8")

            st.caption(
                f"Se han preparado {len(figures_to_export)} gráficos. "
                "Si PNG falla en Streamlit Cloud, descarga el ZIP completo, descomprímelo y ejecuta Arrancar_Convertidor.bat. Ese BAT arranca Streamlit localmente en http://localhost:8501/ y captura PNG."
            )
            if not index_graphs.empty:
                st.dataframe(index_graphs, use_container_width=True)

            with zipfile.ZipFile(zipf, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(xlsx, arcname=xlsx.name)
                z.write(csv, arcname=csv.name)
                z.write(p_report_md, arcname=p_report_md.name)
                z.write(p_report_html, arcname=p_report_html.name)

                # Añadir gráficos exportados, con subcarpetas html/png/svg
                if graphs_dir.exists():
                    for p in graphs_dir.rglob("*"):
                        if p.is_file():
                            z.write(p, arcname=f"graficos/{p.relative_to(graphs_dir)}")

                # Añadir script local de conversión HTML -> PNG
                if converter_script.exists():
                    z.write(converter_script, arcname=converter_script.name)

                # Añadir script local para capturar http://localhost:8501/ como PNG
                if localhost_capture_script.exists():
                    z.write(localhost_capture_script, arcname=localhost_capture_script.name)

                # Añadir arrancador universal Windows
                if arrancador_bat.exists():
                    z.write(arrancador_bat, arcname=arrancador_bat.name)

            st.download_button("Descargar ZIP completo con gráficos", zipf.read_bytes(), file_name="resultados_hrv_comparativa_con_graficos.zip", mime="application/zip")

            # ZIP independiente sólo con PNG
            png_zipf = tmpdir / "graficos_png.zip"
            with zipfile.ZipFile(png_zipf, "w", zipfile.ZIP_DEFLATED) as zpng:
                png_root = graphs_dir / "png"
                if png_root.exists():
                    for p in png_root.rglob("*.png"):
                        zpng.write(p, arcname=p.name)
                # También incluye PNG generados desde HTML localmente si existen
                png_from_html = graphs_dir / "png_from_html"
                if png_from_html.exists():
                    for p in png_from_html.rglob("*.png"):
                        zpng.write(p, arcname=p.name)

            if png_zipf.exists() and png_zipf.stat().st_size > 100:
                st.download_button(
                    "Descargar sólo gráficos PNG",
                    png_zipf.read_bytes(),
                    file_name="graficos_hrv_png.zip",
                    mime="application/zip"
                )
            else:
                st.warning("No se han generado PNG directamente en Streamlit. Descarga el ZIP completo y usa convert_html_to_png.py para convertir los HTML a PNG en tu ordenador.")
            st.download_button("Descargar Excel", xlsx.read_bytes(), file_name="resultados_hrv_comparativa.xlsx")
