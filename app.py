# ============================================================
# app.py — ALL PACE TABLES 1A–2C, 3–6, 7–11 IN ONE APP
# 3 PROFESSIONAL PDFs | LOGIC PRESERVED | WHITE UI
# ============================================================

import io
import math
import numpy as np
import pandas as pd
import streamlit as st

from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="PACE Passenger & VAN Mileage Estimator",
    layout="wide",
    page_icon="🚌",
)

# ============================================================
# GLOBAL APP STYLING (WHITE, PROFESSIONAL)
# ============================================================
st.markdown(
    """
<style>
/* Overall background */
html, body, [class*="css"]  {
    background-color: #f5f7fb;
}

/* Main container padding */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Header row with logo + text */
.pace-header-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 0.3rem;
}

/* Bigger PACE logo */
.pace-logo-circle {
    width: 50px;      /* increased size */
    height: 50px;
    border-radius: 999px;
    border: 5px solid #1458a5;   /* deep PACE-style blue */
    position: relative;
    box-sizing: border-box;
}

.pace-logo-square {
    width: 16px;
    height: 16px;
    background-color: #003f87;   /* darker inner blue */
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}

.pace-logo-tail {
    position: absolute;
    bottom: -4px;
    right: -2px;
    width: 20px;
    height: 20px;
    border-radius: 4px;
    background: #1458a5;
    clip-path: polygon(0 0, 100% 0, 100% 100%);
}

/* Wordmark "pace" — now deeper blue */
.pace-logo-text {
    font-size: 40px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: lowercase;
    color: #1458a5;   /* ✨ stronger blue shade */
}

/* Subtitle under header */
.pace-subtitle {
    font-size: 14px;
    color: #6b7280;
    margin-bottom: 1.0rem;
}

/* Hero / info card */
.pace-hero {
    background-color: #0f172a;
    color: #e5e7eb;
    border-radius: 18px;
    padding: 14px 18px;
    margin-bottom: 1.4rem;
    border: 1px solid #020617;
    font-size: 13px;
}

.pace-hero-title {
    font-weight: 600;
    margin-bottom: 4px;
}

/* Section titles and captions */
.pace-section-title {
    font-size: 18px;
    font-weight: 700;
    color: #111827;
    margin-top: 0.2rem;
    margin-bottom: 0.4rem;
}

.pace-section-caption {
    font-size: 13px;
    color: #6b7280;
    margin-bottom: 0.6rem;
}

/* Cards behind tables if needed */
.pace-card {
    background-color: #ffffff;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.55);
    box-shadow: 0 10px 25px rgba(15,23,42,0.06);
    padding: 0.8rem 1.0rem;
    margin-bottom: 0.9rem;
}

/* DataFrame container */
[data-testid="stDataFrame"] {
    background-color: #ffffff;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.45);
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 14px;
    font-weight: 600;
}

/* Download buttons */
.stDownloadButton button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    padding: 7px 16px;
    border-radius: 999px;
    font-weight: 600;
    border: none;
}
.stDownloadButton button:hover {
    background: linear-gradient(90deg, #1d4ed8, #1e40af);
    color: #f9fafb;
}

/* Info / spinner / success boxes on white */
[data-testid="stAlert"] {
    border-radius: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HEADER WITH PACE-STYLE LOGO
# ============================================================
st.markdown(
    """
<div class="pace-header-row">
  <div style="position:relative; display:flex; align-items:center; justify-content:center;">
    <div class="pace-logo-circle">
      <div class="pace-logo-square"></div>
      <div class="pace-logo-tail"></div>
    </div>
  </div>
  <div class="pace-logo-text">pace</div>
</div>
<div class="pace-subtitle">
Passenger Miles & VAN Mileage Estimates — Official Tables 1A–2C, 3–6, 7–11
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="pace-hero">
  <div class="pace-hero-title">What this tool generates</div>
  <ul style="margin-top:2px; margin-bottom:4px;">
    <li><b>Tables 1A–1C, 2A–2C</b> — PACE-owned & Contract Garages</li>
    <li><b>Tables 3–6</b> — Suburban ADA / DAR & Chicago ADA</li>
    <li><b>Tables 7–11</b> — VAN Passenger Miles, Miles & Hours</li>
  </ul>
  <div style="font-size:12px; opacity:0.9; margin-top:4px;">
    Upload the appropriate input files in each tab and export consolidated PDFs for reporting.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
DISPLAY_ORDER = [
    "Fox valley", "Heritage", "North", "North Shore", "Northwest",
    "River", "South", "Southwest", "West",
]

CONTRACT_2A = ["First Student", "Highland Park", "MV Batavia", "Niles"]
CONTRACT_2B = ["First Student", "Highland Park", "Niles"]
CONTRACT_2C = ["First Student", "Niles"]

PASSENGERS_ACTUAL = {
    "Weekday": {
        "Fox valley": 281091,
        "Heritage": 683365,
        "North": 991615,
        "North Shore": 748987,
        "Northwest": 2993804,
        "River": 750379,
        "South": 2414196,
        "Southwest": 1322351,
        "West": 3791146,
        "First Student": 8000,
        "Highland Park": 43991,
        "MV Batavia": 95976,
        "Niles": 99676,
    },
    "Saturday": {
        "Fox valley": 35976,
        "Heritage": 35803,
        "North": 70407,
        "North Shore": 61072,
        "Northwest": 343061,
        "River": 81936,
        "South": 323437,
        "Southwest": 132363,
        "West": 424081,
        "First Student": 7405,
        "Highland Park": 24932,
        "Niles": 18655,
    },
    "Sunday": {
        "North": 25417,
        "Northwest": 310538,
        "South": 222494,
        "Southwest": 81751,
        "West": 275721,
        "First Student": 3527,
        "Niles": 14450,
    },
}

ACTUAL_ADA_DAR = {
    "Sub ADA": {"Weekday": 493853, "Saturday": 41863, "Sunday": 22100},
    "Sub DAR": {"Weekday": 294376, "Saturday": 16255, "Sunday": 8185},
    "Chicago ADA": {"Weekday": 1617652, "Saturday": 169419, "Sunday": 197330},
}

SERVICES_VAN = ["ADA", "MUNICIPAL", "SHUTTLE", "VIP"]

WORKDAYS_VAN = {
    "ADA": 27750,
    "MUNICIPAL": 13512,
    "SHUTTLE": 3616,
    "VIP": 21892,
}

PASSENGERS_ACTUAL_VAN = {
    "ADA": 288_527,
    "MUNICIPAL": 149_568,
    "SHUTTLE": 33_595,
    "VIP": 211_786,
}

# ============================================================
# SHARED HELPERS
# ============================================================
def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def export_multi_table_pdf(main_title: str, tables):
    """
    Generic multi-table PDF for Tables 1A–2C and 3–6.
    tables = [(subtitle, df), ...]
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(letter),
        leftMargin=28,
        rightMargin=28,
        topMargin=28,
        bottomMargin=28,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "TitleMain", fontSize=16, alignment=TA_CENTER, spaceAfter=14
        )
    )
    styles.add(
        ParagraphStyle(
            "TitleTable", fontSize=11, alignment=TA_LEFT, spaceAfter=6
        )
    )

    elements = [Paragraph(main_title, styles["TitleMain"])]

    for t_title, df in tables:
        elements.append(Paragraph(t_title, styles["TitleTable"]))
        data = [df.columns.tolist()]
        for _, r in df.iterrows():
            row = []
            for v in r:
                if isinstance(v, (int, np.integer)):
                    row.append(f"{v:,}")
                elif isinstance(v, float):
                    row.append(f"{v:,.3f}")
                else:
                    row.append(str(v))
            data.append(row)

        tbl = Table(data, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    ("ALIGN", (0, 1), (0, -1), "LEFT"),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ]
            )
        )

        elements.extend([tbl, Spacer(1, 10)])

    doc.build(elements)
    return buf.getvalue()


# ---------- Generic display formatter with commas ----------
def format_with_commas(df: pd.DataFrame) -> pd.DataFrame:
    """
    For display only: adds commas to integer columns and formats floats as #,###.000.
    Used for all Streamlit dataframes.
    """
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_integer_dtype(d[c]):
            d[c] = d[c].map(lambda x: f"{x:,}")
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: f"{x:,.3f}")
    return d

# ============================================================
# SECTION 1 — TABLES 1A–2C (LOGIC PRESERVED)
# ============================================================
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    rename_map = {
        "PASSENGERS": "PASSENGER",
        "Passengers": "PASSENGER",
        "Passenger": "PASSENGER",
        "PASSENGERS_OFF": "PASSENGER",
        "PASSENGER MILES": "PASSENGER_MILES",
        "Passenger Miles": "PASSENGER_MILES",
        "PASSENGER Miles": "PASSENGER_MILES",
        "GARAGE": "GARAGE_NAME",
        "Garage": "GARAGE_NAME",
        "DIVISION": "GARAGE_NAME",
        "Division": "GARAGE_NAME",
        "SERVICE PERIOD": "SERVICE_PERIOD",
        "Service Period": "SERVICE_PERIOD",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    return df


def canonical_division(x):
    if not isinstance(x, str):
        return x

    s = x.strip().lower()
    mapping = {
        "fox valley": "Fox valley",
        "heritage": "Heritage",
        "north": "North",
        "north shore": "North Shore",
        "northshore": "North Shore",
        "north-west": "Northwest",
        "northwest": "Northwest",
        "river": "River",
        "south": "South",
        "southwest": "Southwest",
        "south west": "Southwest",
        "west": "West",
        "first student": "First Student",
        "highland park": "Highland Park",
        "mv batavia": "MV Batavia",
        "mvbatavia": "MV Batavia",
        "mvbatavia ": "MV Batavia",
        "niles": "Niles",
        "village of niles": "Niles",
    }
    return mapping.get(s, x)


def load_all_1x2(xls_bytes: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(xls_bytes))
    frames = []

    for sh in xls.sheet_names:
        df = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=sh)
        df = clean_cols(df)

        if {"SERVICE_PERIOD", "PASSENGER", "PASSENGER_MILES"}.issubset(df.columns):
            if "GARAGE_NAME" not in df.columns:
                df["GARAGE_NAME"] = sh
            frames.append(df)

    if not frames:
        raise ValueError("No valid sheets found for Tables 1A–2C.")

    df_all = pd.concat(frames, ignore_index=True)
    df_all["GARAGE_NAME"] = df_all["GARAGE_NAME"].map(canonical_division)
    return df_all


def compute_table_1x2(df_all, day, contract=False, allowed_services=None):
    df = df_all[df_all["SERVICE_PERIOD"].str.lower() == day.lower()].copy()

    df["PASSENGER"] = pd.to_numeric(df["PASSENGER"], errors="coerce")
    df["PASSENGER_MILES"] = pd.to_numeric(df["PASSENGER_MILES"], errors="coerce")

    if contract:
        df = df[df["GARAGE_NAME"].isin(allowed_services)]
    else:
        df = df[df["GARAGE_NAME"].isin(DISPLAY_ORDER)]

    sampled = df[df["PASSENGER"] > 0]

    rows = []
    for div, g in sampled.groupby("GARAGE_NAME"):
        if div not in PASSENGERS_ACTUAL[day]:
            continue

        n = len(g)
        total_pass = g["PASSENGER"].sum()
        total_miles = g["PASSENGER_MILES"].sum()
        mean = total_miles / total_pass if total_pass else np.nan

        if n > 1:
            g = g.copy()
            g["r_i"] = g["PASSENGER_MILES"] / g["PASSENGER"]
            sd = math.sqrt(
                (1 / (n - 1))
                * ((g["PASSENGER"] * (g["r_i"] - mean) ** 2).sum() / total_pass)
            )
        else:
            sd = np.nan

        pax_act = PASSENGERS_ACTUAL[day][div]

        rows.append(
            {
                "Service": div,
                "Runs": n,
                "Passengers sampled": int(total_pass),
                "Passengers actual": pax_act,
                "Estimated avg miles": mean,
                "Estimated total miles": round_half_up(mean * pax_act),
                "SE": sd,
                "Precision": (2 * sd / mean) if (mean > 0 and sd == sd) else np.nan,
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    if contract:
        out["Service"] = pd.Categorical(
            out["Service"],
            CONTRACT_2A
            if day == "Weekday"
            else (CONTRACT_2B if day == "Saturday" else CONTRACT_2C),
            ordered=True,
        )
    else:
        out["Service"] = pd.Categorical(out["Service"], DISPLAY_ORDER, ordered=True)

    out = out.sort_values("Service").reset_index(drop=True)

    Xk = out["Passengers actual"]
    rk = out["Estimated avg miles"]
    SEk = out["SE"]

    overall_avg = (Xk * rk).sum() / Xk.sum()
    overall_SE = math.sqrt(((Xk**2) * (SEk**2)).sum()) / Xk.sum()
    overall_prec = 2 * overall_SE / overall_avg
    overall_tot = int(out["Estimated total miles"].sum())

    overall = pd.DataFrame(
        [
            {
                "Service": "Overall",
                "Runs": out["Runs"].sum(),
                "Passengers sampled": out["Passengers sampled"].sum(),
                "Passengers actual": Xk.sum(),
                "Estimated avg miles": overall_avg,
                "Estimated total miles": overall_tot,
                "SE": overall_SE,
                "Precision": overall_prec,
            }
        ]
    )

    final = pd.concat([out, overall], ignore_index=True)

    final["Estimated avg miles"] = final["Estimated avg miles"].round(3)
    final["Estimated total miles"] = final["Estimated total miles"].round(0)
    final["SE"] = final["SE"].round(3)
    final["Precision"] = final["Precision"].round(3)

    return final

# ============================================================
# SECTION 2 — TABLES 3–6 (ADA / DAR) — LOGIC PRESERVED
# ============================================================
def classify_day(row):
    dow = str(row["DOW"]).lower()
    if dow.startswith("sat"):
        return "Saturday"
    if dow.startswith("sun"):
        return "Sunday"
    return "Weekday"


def load_ada_data(xls_bytes):
    xls = pd.ExcelFile(io.BytesIO(xls_bytes))
    sheets = ["Sub_ADA", "Sub_DAR", "Chicago_ADA"]
    data = {}

    for sh in sheets:
        df = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=sh)
        df.columns = df.columns.str.strip()

        df["Rides"] = pd.to_numeric(df["Rides"], errors="coerce").fillna(0)
        df["Miles"] = pd.to_numeric(df["Miles"], errors="coerce").fillna(0)
        df["day_type"] = df.apply(classify_day, axis=1)

        data[sh] = df

    return data


def compute_single_table(df, actual_dict):
    rows = []
    for day in ["Weekday", "Saturday", "Sunday"]:
        g = df[df["day_type"] == day].copy()
        runs = len(g)

        if len(g) == 0:
            rows.append(
                {
                    "Service": day,
                    "Runs": runs,
                    "Passengers sampled": 0,
                    "Passengers actual": actual_dict[day],
                    "Estimated avg miles": np.nan,
                    "Estimated total miles": np.nan,
                    "SE": np.nan,
                    "Precision": np.nan,
                }
            )
            continue

        g = g[g["Rides"] > 0]

        total_pass = g["Rides"].sum()
        total_miles = g["Miles"].sum()

        mean = total_miles / total_pass if total_pass > 0 else np.nan

        n = len(g)
        if n > 1:
            g["r_i"] = g["Miles"] / g["Rides"]
            var_num = (g["Rides"] * (g["r_i"] - mean) ** 2).sum()
            sd = math.sqrt((1 / (n - 1)) * (var_num / total_pass))
        else:
            sd = np.nan

        act = actual_dict[day]
        est_total = round_half_up(mean * act)
        precision = (2 * sd / mean) if (mean > 0 and sd == sd) else np.nan

        rows.append(
            {
                "Service": day,
                "Runs": runs,
                "Passengers sampled": int(total_pass),
                "Passengers actual": act,
                "Estimated avg miles": mean,
                "Estimated total miles": est_total,
                "SE": sd,
                "Precision": precision,
            }
        )

    df_rows = pd.DataFrame(rows)

    valid = df_rows.dropna(subset=["Estimated avg miles"])
    X = valid["Passengers actual"]
    r = valid["Estimated avg miles"]
    SE = valid["SE"]

    overall_avg = (X * r).sum() / X.sum()
    overall_SE = math.sqrt(((X**2) * (SE**2)).sum()) / X.sum()
    overall_precision = 2 * overall_SE / overall_avg
    overall_total = round_half_up(overall_avg * X.sum())

    overall = pd.DataFrame(
        [
            {
                "Service": "Overall",
                "Runs": valid["Runs"].sum(),
                "Passengers sampled": valid["Passengers sampled"].sum(),
                "Passengers actual": int(X.sum()),
                "Estimated avg miles": overall_avg,
                "Estimated total miles": overall_total,
                "SE": overall_SE,
                "Precision": overall_precision,
            }
        ]
    )

    final = pd.concat([df_rows, overall], ignore_index=True)

    for col in ["Estimated avg miles", "SE", "Precision"]:
        final[col] = final[col].round(3)
    final["Estimated total miles"] = final["Estimated total miles"].round(0)

    return final


def compute_table5(table3, table4):
    rows = []

    for day in ["Weekday", "Saturday", "Sunday"]:
        r3 = table3[table3["Service"] == day].iloc[0]
        r4 = table4[table4["Service"] == day].iloc[0]

        X3 = r3["Passengers actual"]
        X4 = r4["Passengers actual"]
        X = X3 + X4

        avg = (X3 * r3["Estimated avg miles"] + X4 * r4["Estimated avg miles"]) / X

        SE = math.sqrt((X3**2 * r3["SE"]**2 + X4**2 * r4["SE"]**2)) / X
        precision = 2 * SE / avg

        est_total = r3["Estimated total miles"] + r4["Estimated total miles"]

        rows.append(
            {
                "Service": day,
                "Runs": r3["Runs"] + r4["Runs"],
                "Passengers sampled": r3["Passengers sampled"]
                + r4["Passengers sampled"],
                "Passengers actual": X,
                "Estimated avg miles": avg,
                "Estimated total miles": est_total,
                "SE": SE,
                "Precision": precision,
            }
        )

    df_rows = pd.DataFrame(rows)

    overall = {
        "Service": "Overall",
        "Runs": df_rows["Runs"].sum(),
        "Passengers sampled": df_rows["Passengers sampled"].sum(),
        "Passengers actual": df_rows["Passengers actual"].sum(),
        "Estimated avg miles": (df_rows["Passengers actual"] * df_rows["Estimated avg miles"]).sum()
        / df_rows["Passengers actual"].sum(),
        "Estimated total miles": df_rows["Estimated total miles"].sum(),
        "SE": math.sqrt(((df_rows["Passengers actual"] ** 2) * (df_rows["SE"] ** 2)).sum())
        / df_rows["Passengers actual"].sum(),
    }

    overall["Precision"] = 2 * overall["SE"] / overall["Estimated avg miles"]

    df_rows = pd.concat([df_rows, pd.DataFrame([overall])], ignore_index=True)

    df_rows["Estimated avg miles"] = df_rows["Estimated avg miles"].round(3)
    df_rows["Estimated total miles"] = df_rows["Estimated total miles"].round(0)
    df_rows["SE"] = df_rows["SE"].round(3)
    df_rows["Precision"] = df_rows["Precision"].round(3)

    return df_rows

# ============================================================
# SECTION 3 — VAN TABLES 7–11 (LOGIC PRESERVED)
# ============================================================
def format_for_display_van(df):
    # just reuse generic formatter
    return format_with_commas(df)


def load_van_data(csv_bytes):
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.columns = df.columns.str.strip()

    df["SERVICE"] = df["DIVISION"].astype(str).str.upper().str.strip()

    df["REV_MILES"] = pd.to_numeric(df["trip_dist"], errors="coerce")
    df["ACT_MILES"] = df["REV_MILES"] + pd.to_numeric(
        df["Dhead.mil"], errors="coerce"
    ).fillna(0)

    df["REV_HOURS"] = pd.to_numeric(df["time_trip"], errors="coerce") / 60
    df["ACT_HOURS"] = (
        pd.to_numeric(df["time_trip"], errors="coerce")
        + pd.to_numeric(df["dhead_time"], errors="coerce").fillna(0)
    ) / 60

    df["TOT_BOARD"] = pd.to_numeric(df["TOT_BOARD"], errors="coerce")
    df["tot.pas_mil"] = pd.to_numeric(df["tot.pas_mil"], errors="coerce")

    return df


def table7_passenger_miles(df):
    rows, rk, SEk, Xk = [], [], [], []

    for svc in SERVICES_VAN:
        g = df[df["SERVICE"] == svc]
        n = len(g)
        X = PASSENGERS_ACTUAL_VAN[svc]

        sampled = g["TOT_BOARD"].sum()
        mean = g["tot.pas_mil"].sum() / sampled

        r_i = g["tot.pas_mil"] / g["TOT_BOARD"]
        se = math.sqrt(
            (1 / (n - 1)) * ((g["TOT_BOARD"] * (r_i - mean) ** 2).sum() / sampled)
        )

        rows.append(
            {
                "Service": svc.title(),
                "Runs": n,
                "Passengers Sampled": int(sampled),
                "Passengers Actual": X,
                "Est. Avg Miles": mean,
                "Est. Total Miles": round_half_up(mean * X),
                "SE": se,
                "Precision": 2 * se / mean,
            }
        )

        rk.append(mean)
        SEk.append(se)
        Xk.append(X)

    rk, SEk, Xk = map(np.array, (rk, SEk, Xk))
    avg_o = (rk * Xk).sum() / Xk.sum()
    se_o = math.sqrt((Xk ** 2 * SEk ** 2).sum()) / Xk.sum()

    rows.append(
        {
            "Service": "Overall",
            "Runs": len(df),
            "Passengers Sampled": int(df["TOT_BOARD"].sum()),
            "Passengers Actual": int(Xk.sum()),
            "Est. Avg Miles": avg_o,
            "Est. Total Miles": round_half_up(avg_o * Xk.sum()),
            "SE": se_o,
            "Precision": 2 * se_o / avg_o,
        }
    )

    t = pd.DataFrame(rows)
    t.iloc[:, 4:] = t.iloc[:, 4:].round(3)
    return t


def compute_van_table(df, col):
    rows, rk, SEk, Xk = [], [], [], []

    for svc in SERVICES_VAN:
        x = df[df["SERVICE"] == svc][col].dropna().to_numpy()
        n = len(x)
        X = WORKDAYS_VAN[svc]

        mean = x.mean()
        se = x.std(ddof=1) / math.sqrt(n - 1)

        rows.append(
            {
                "Service": svc.title(),
                "Runs": n,
                "Workdays": X,
                "Est. Avg": mean,
                "SE": se,
                "Est. Total": round_half_up(mean * X),
                "Half-Width": round_half_up(2 * X * se),
                "Precision": 2 * se / mean,
            }
        )

        rk.append(mean)
        SEk.append(se)
        Xk.append(X)

    rk, SEk, Xk = map(np.array, (rk, SEk, Xk))
    avg_o = (rk * Xk).sum() / Xk.sum()
    se_o = math.sqrt((Xk ** 2 * SEk ** 2).sum()) / Xk.sum()

    rows.append(
        {
            "Service": "Overall",
            "Runs": len(df),
            "Workdays": int(Xk.sum()),
            "Est. Avg": avg_o,
            "SE": se_o,
            "Est. Total": round_half_up(avg_o * Xk.sum()),
            "Half-Width": round_half_up(2 * Xk.sum() * se_o),
            "Precision": 2 * se_o / avg_o,
        }
    )

    t = pd.DataFrame(rows)
    t.iloc[:, 3:] = t.iloc[:, 3:].round(3)
    return t


def export_single_page_pdf_van(tables):
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(letter),
        leftMargin=28,
        rightMargin=28,
        topMargin=28,
        bottomMargin=28,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "MainTitle", fontSize=16, alignment=TA_CENTER, spaceAfter=10
        )
    )
    styles.add(
        ParagraphStyle(
            "TableTitle", fontSize=11, alignment=TA_LEFT, spaceAfter=6
        )
    )

    elements = [
        Paragraph(
            "PACE – VAN Mileage & Hours Estimates (Tables 7–11)",
            styles["MainTitle"],
        )
    ]

    COL_WIDTHS = [70, 45, 75, 70, 55, 80, 80, 60]

    for title, df in tables:
        elements.append(Paragraph(title, styles["TableTitle"]))

        data = [df.columns.tolist()]
        for _, r in df.iterrows():
            row = []
            for v in r:
                if isinstance(v, (int, np.integer)):
                    row.append(f"{v:,}")
                elif isinstance(v, float):
                    row.append(f"{v:,.3f}")
                else:
                    row.append(str(v))
            data.append(row)

        tbl = Table(data, colWidths=COL_WIDTHS, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (0, 1), (0, -1), "LEFT"),
                    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        elements.append(tbl)
        elements.append(Spacer(1, 10))

    doc.build(elements)
    return buf.getvalue()

# ============================================================
# UI LAYOUT — TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(
    [
        "Tables 1A–2C (PACE Divisions & Contracts)",
        "Tables 3–6 (ADA / DAR)",
        "Tables 7–11 (VAN Services)",
    ]
)

# ---------------- TAB 1 ----------------
with tab1:
    st.markdown(
        "<div class='pace-section-title'>Tables 1A–1C & 2A–2C — PACE Divisions & Contract Carriers</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='pace-section-caption'>Upload the Excel workbook with "
        "<b>SERVICE_PERIOD</b>, <b>PASSENGER</b>, <b>PASSENGER_MILES</b>, and "
        "<b>GARAGE_NAME</b> (or equivalent columns).</div>",
        unsafe_allow_html=True,
    )

    up1 = st.file_uploader(
        "Upload Excel for Tables 1A–2C",
        type=["xlsx"],
        key="up_1x2",
    )

    if up1 is None:
        st.info("⬅️ Upload the Excel file to generate Tables 1A–2C.")
    else:
        try:
            df_all = load_all_1x2(up1.read())

            with st.spinner("Computing Tables 1A–2C ..."):
                t1a = compute_table_1x2(df_all, "Weekday")
                t1b = compute_table_1x2(df_all, "Saturday")
                t1c = compute_table_1x2(df_all, "Sunday")

                t2a = compute_table_1x2(
                    df_all,
                    "Weekday",
                    contract=True,
                    allowed_services=CONTRACT_2A,
                )
                t2b = compute_table_1x2(
                    df_all,
                    "Saturday",
                    contract=True,
                    allowed_services=CONTRACT_2B,
                )
                t2c = compute_table_1x2(
                    df_all,
                    "Sunday",
                    contract=True,
                    allowed_services=CONTRACT_2C,
                )

            st.success("✨ Tables 1A–2C generated!")

            tables_1x2 = [
                ("Table 1A (Weekday)", t1a),
                ("Table 1B (Saturday)", t1b),
                ("Table 1C (Sunday)", t1c),
                ("Table 2A (Weekday — Contract)", t2a),
                ("Table 2B (Saturday — Contract)", t2b),
                ("Table 2C (Sunday — Contract)", t2c),
            ]

            for name, df in tables_1x2:
                st.markdown(
                    f"<div class='pace-card'><div class='pace-section-title' style='font-size:15px'>{name}</div>",
                    unsafe_allow_html=True,
                )
                # 🔹 Display with commas
                st.dataframe(format_with_commas(df), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            pdf_1x2 = export_multi_table_pdf(
                "PACE — Tables 1A–2C (PACE Divisions & Contract Carriers)",
                tables_1x2,
            )

            st.markdown("#### Export")
            st.download_button(
                "⬇️ Download Tables 1A–2C (Single PDF)",
                pdf_1x2,
                "PACE_Tables_1A_2C.pdf",
                "application/pdf",
            )

        except Exception as e:
            st.error("Processing error for Tables 1A–2C:")
            st.exception(e)

# ---------------- TAB 2 ----------------
with tab2:
    st.markdown(
        "<div class='pace-section-title'>Tables 3–6 — ADA / DAR Passenger Miles</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='pace-section-caption'>Upload <b>cleaned_data.xlsx</b> with sheets "
        "<b>Sub_ADA</b>, <b>Sub_DAR</b>, and <b>Chicago_ADA</b> and columns "
        "<b>Date</b>, <b>Service</b>, <b>Rides</b>, <b>Miles</b>, <b>DOW</b>.</div>",
        unsafe_allow_html=True,
    )

    up2 = st.file_uploader(
        "Upload cleaned_data.xlsx for ADA/DAR Tables 3–6",
        type=["xlsx"],
        key="up_3x6",
    )

    if up2 is None:
        st.info("⬅️ Upload cleaned_data.xlsx to generate Tables 3–6.")
    else:
        try:
            data = load_ada_data(up2.read())
            df_sub_ada = data["Sub_ADA"]
            df_sub_dar = data["Sub_DAR"]
            df_chi_ada = data["Chicago_ADA"]

            with st.spinner("Computing Tables 3–6 ..."):
                table3 = compute_single_table(
                    df_sub_ada, ACTUAL_ADA_DAR["Sub ADA"]
                )
                table4 = compute_single_table(
                    df_sub_dar, ACTUAL_ADA_DAR["Sub DAR"]
                )
                table5 = compute_table5(table3, table4)
                table6 = compute_single_table(
                    df_chi_ada, ACTUAL_ADA_DAR["Chicago ADA"]
                )

            st.success("✨ Tables 3–6 computed successfully!")

            tables_3_6 = [
                ("Table 3 — Suburban ADA", table3),
                ("Table 4 — Suburban DAR", table4),
                ("Table 5 — Suburban Total", table5),
                ("Table 6 — Chicago ADA", table6),
            ]

            for name, df in tables_3_6:
                st.markdown(
                    f"<div class='pace-card'><div class='pace-section-title' style='font-size:15px'>{name}</div>",
                    unsafe_allow_html=True,
                )
                # 🔹 Display with commas
                st.dataframe(format_with_commas(df), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            pdf_3_6 = export_multi_table_pdf(
                "PACE — Tables 3–6 (ADA / DAR Passenger Miles)", tables_3_6
            )

            st.markdown("#### Export")
            st.download_button(
                "⬇️ Download Tables 3–6 (Single PDF)",
                pdf_3_6,
                "PACE_Tables_3_6_ADA_DAR.pdf",
                "application/pdf",
            )

        except Exception as e:
            st.error("Error while processing Tables 3–6:")
            st.exception(e)

# ---------------- TAB 3 ----------------
with tab3:
    st.markdown(
        "<div class='pace-section-title'>Tables 7–11 — VAN Mileage & Hours Estimates</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='pace-section-caption'>Upload <b>cleaned_data.csv</b> for VAN services with "
        "columns <b>DIVISION</b>, <b>trip_dist</b>, <b>Dhead.mil</b>, <b>time_trip</b>, "
        "<b>dhead_time</b>, <b>TOT_BOARD</b>, <b>tot.pas_mil</b>.</div>",
        unsafe_allow_html=True,
    )

    up3 = st.file_uploader(
        "Upload cleaned_data.csv for VAN Tables 7–11",
        type=["csv"],
        key="up_7x11",
    )

    if up3 is None:
        st.info("⬅️ Upload cleaned_data.csv to generate Tables 7–11.")
    else:
        try:
            df_van = load_van_data(up3.read())

            with st.spinner("Computing VAN Tables 7–11 ..."):
                t7 = table7_passenger_miles(df_van)
                t8 = compute_van_table(df_van, "ACT_MILES")
                t9 = compute_van_table(df_van, "REV_MILES")
                t10 = compute_van_table(df_van, "ACT_HOURS")
                t11 = compute_van_table(df_van, "REV_HOURS")

            st.success("✨ VAN Tables 7–11 computed!")

            tables_7_11 = [
                ("Table 7 — Passenger Miles Estimates for Van Services", t7),
                ("Table 8 — Actual Miles Estimates for Van Services", t8),
                ("Table 9 — Revenue Miles Estimates for Van Services", t9),
                ("Table 10 — Actual Hours Estimates for Van Services", t10),
                ("Table 11 — Revenue Hours Estimates for Van Services", t11),
            ]

            for name, df in tables_7_11:
                st.markdown(
                    f"<div class='pace-card'><div class='pace-section-title' style='font-size:15px'>{name}</div>",
                    unsafe_allow_html=True,
                )
                # already formats with commas
                st.dataframe(format_for_display_van(df), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            pdf_van = export_single_page_pdf_van(
                [
                    (
                        "Table 7. Passenger Miles Estimates for Van Services",
                        t7,
                    ),
                    ("Table 8. Actual Miles Estimates for Van Services", t8),
                    ("Table 9. Revenue Miles Estimates for Van Services", t9),
                    ("Table 10. Actual Hours Estimates for Van Services", t10),
                    ("Table 11. Revenue Hours Estimates for Van Services", t11),
                ]
            )

            st.markdown("#### Export")
            st.download_button(
                "⬇️ Download VAN Tables 7–11 Professional PDF (Single Page)",
                pdf_van,
                "PACE_VAN_Tables_7_to_11.pdf",
                "application/pdf",
            )

        except Exception as e:
            st.error("Error while processing VAN Tables 7–11:")
            st.exception(e)
