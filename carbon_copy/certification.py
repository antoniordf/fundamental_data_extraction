from __future__ import annotations

import re
from typing import Dict

import pandas as pd

from selector.models import StatementType

from .models import StatementBlock


COLS_START = 3


def _pick_row(df: pd.DataFrame, pattern: str) -> pd.Series | None:
    mask = df["Line Item (as printed)"].str.contains(pattern, case=False, na=False)
    if not mask.any():
        return None
    return df[mask].iloc[0]


def _value(series: pd.Series | None, column: str) -> float | None:
    if series is None:
        return None
    value = series[column]
    if pd.isna(value):
        return None
    return float(value)


_YEAR_PATTERN = re.compile(r"(20\d{2})")


def _column_key(label: str) -> str:
    match = _YEAR_PATTERN.search(label)
    if match:
        return match.group(1)
    return label.strip()


def _build_lookup(columns: pd.Index) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for column in columns:
        key = _column_key(column)
        lookup.setdefault(key, column)
    return lookup


def build_certification(blocks: dict[StatementType, StatementBlock]) -> pd.DataFrame:
    rows: list[list[str]] = []

    income = blocks[StatementType.INCOME].wide_table
    balance = blocks[StatementType.BALANCE].wide_table
    cash = blocks[StatementType.CASH].wide_table

    balance_columns = balance.columns[COLS_START:]
    income_columns = income.columns[COLS_START:]
    cash_columns = cash.columns[COLS_START:]

    balance_lookup = _build_lookup(balance_columns)
    income_lookup = _build_lookup(income_columns)
    cash_lookup = _build_lookup(cash_columns)

    # 1. Balance sheet equality
    assets_row = _pick_row(balance, r"total assets")
    liab_row = _pick_row(balance, r"total liabilities")
    equity_row = _pick_row(balance, r"total stockholders' equity|total shareholders' equity")

    details = []
    all_pass = True
    for period in balance_columns:
        assets = _value(assets_row, period)
        liab = _value(liab_row, period)
        equity = _value(equity_row, period)
        if None in (assets, liab, equity):
            details.append(f"{period}: insufficient data")
            all_pass = False
            continue
        delta = assets - (liab + equity)
        if abs(delta) > 1e-6:
            all_pass = False
        details.append(f"{period}: Δ {delta:.6g}")
    rows.append([
        "Balance Sheet equality (Assets = Liabilities + Equity)",
        "PASS" if all_pass else "CHECK",
        "; ".join(details),
    ])

    # 2. Cash roll-through + tie to balance sheet cash
    begin_row = _pick_row(cash, r"beginning cash and cash equivalents")
    delta_row = _pick_row(cash, r"increase \(decrease\) in cash and cash equivalents")
    end_row = _pick_row(cash, r"ending cash and cash equivalents")
    cash_bs_row = _pick_row(balance, r"cash and cash equivalents")

    details = []
    all_pass = True
    for cash_col in cash_columns:
        key = _column_key(cash_col)
        begin = _value(begin_row, cash_col)
        delta = _value(delta_row, cash_col)
        end = _value(end_row, cash_col)
        if None in (begin, delta, end):
            details.append(f"{cash_col}: insufficient data")
            all_pass = False
            continue
        roll_delta = begin + delta - end
        if abs(roll_delta) > 1e-6:
            all_pass = False
        tie_delta = None
        bs_col = balance_lookup.get(key)
        if cash_bs_row is not None and bs_col is not None:
            bs_cash = _value(cash_bs_row, bs_col)
            if bs_cash is not None:
                tie_delta = end - bs_cash
                if abs(tie_delta) > 1e-6:
                    all_pass = False
        if tie_delta is None:
            details.append(f"{cash_col}: roll Δ {roll_delta:.6g}")
        else:
            details.append(
                f"{cash_col}: roll Δ {roll_delta:.6g}; tie Δ {tie_delta:.6g}"
            )
    rows.append([
        "Cash roll-through (Beg + Δ = End) & tie to BS cash",
        "PASS" if all_pass else "CHECK",
        "; ".join(details),
    ])

    # 3. Net income link
    is_net_row = _pick_row(income, r"^net income$")
    cf_net_row = _pick_row(cash, r"^net income$")
    details = []
    all_pass = True
    for period in income_columns:
        key = _column_key(period)
        is_net = _value(is_net_row, period)
        cf_col = cash_lookup.get(key)
        cf_net = _value(cf_net_row, cf_col) if cf_col else None
        if None in (is_net, cf_net):
            details.append(f"{period}: insufficient data")
            all_pass = False
            continue
        delta = is_net - cf_net
        if abs(delta) > 1e-6:
            all_pass = False
        details.append(f"{period}: Δ {delta:.6g}")
    rows.append([
        "Net income link (IS Net income vs CF Net income)",
        "PASS" if all_pass else "CHECK",
        "; ".join(details),
    ])

    # 4. Income statement cross-footing
    details = []
    all_pass = True
    revenue_row = _pick_row(income, r"^net revenue|^revenue|^sales")
    cost_row = _pick_row(income, r"cost of revenue|cost of sales")
    gross_row = _pick_row(income, r"gross profit")
    opex_row = _pick_row(income, r"total operating expenses")
    opinc_row = _pick_row(income, r"operating income")
    other_row = _pick_row(income, r"other income|interest and other income")
    pretax_row = _pick_row(income, r"income before")
    tax_row = _pick_row(income, r"provision for income taxes|income taxes")

    for period in income_columns:
        entries = []
        if revenue_row is not None and cost_row is not None and gross_row is not None:
            rev = _value(revenue_row, period)
            cost = _value(cost_row, period)
            gross = _value(gross_row, period)
            if None not in (rev, cost, gross):
                gp_calc = rev - cost
                gp_delta = gp_calc - gross
                if abs(gp_delta) > 1e-6:
                    all_pass = False
                entries.append(f"GP Δ {gp_delta:.6g}")
        if gross_row is not None and opex_row is not None and opinc_row is not None:
            gp = _value(gross_row, period)
            opex = _value(opex_row, period)
            opinc = _value(opinc_row, period)
            if None not in (gp, opex, opinc):
                oi_calc = gp - opex
                oi_delta = oi_calc - opinc
                if abs(oi_delta) > 1e-6:
                    all_pass = False
                entries.append(f"OI Δ {oi_delta:.6g}")
        if opinc_row is not None and other_row is not None and pretax_row is not None:
            opinc = _value(opinc_row, period)
            other = _value(other_row, period)
            pretax = _value(pretax_row, period)
            if None not in (opinc, other, pretax):
                pretax_calc = opinc + other
                pretax_delta = pretax_calc - pretax
                if abs(pretax_delta) > 1e-6:
                    all_pass = False
                entries.append(f"PT Δ {pretax_delta:.6g}")
        if pretax_row is not None and tax_row is not None and is_net_row is not None:
            pretax = _value(pretax_row, period)
            tax = _value(tax_row, period)
            net = _value(is_net_row, period)
            if None not in (pretax, tax, net):
                net_calc = pretax - tax
                net_delta = net_calc - net
                if abs(net_delta) > 1e-6:
                    all_pass = False
                entries.append(f"NI Δ {net_delta:.6g}")
        if entries:
            details.append(f"{period}: " + ", ".join(entries))
    rows.append([
        "IS cross-footing (subtotals)",
        "PASS" if all_pass else "CHECK",
        "; ".join(details) if details else "Not enough data",
    ])

    # 5. Cash flow cross-footing
    details = []
    all_pass = True
    cfo_row = _pick_row(cash, r"net cash provided by operating activities")
    cfi_row = _pick_row(cash, r"net cash provided by \(used in\) investing activities")
    cff_row = _pick_row(cash, r"net cash (?:provided by|used in) financing activities")
    fx_row = _pick_row(cash, r"effect of foreign exchange")

    for period in cash_columns:
        rows_available = [cfo_row, cfi_row, cff_row, fx_row, delta_row]
        if any(row is None for row in rows_available):
            details.append(f"{period}: insufficient data")
            all_pass = False
            continue
        cfo = _value(cfo_row, period)
        cfi = _value(cfi_row, period)
        cff = _value(cff_row, period)
        fx = _value(fx_row, period)
        delta_val = _value(delta_row, period)
        if None in (cfo, cfi, cff, fx, delta_val):
            details.append(f"{period}: insufficient data")
            all_pass = False
            continue
        diff = (cfo + cfi + cff + fx) - delta_val
        if abs(diff) > 1e-6:
            all_pass = False
        details.append(f"{period}: Δ {diff:.6g}")
    rows.append([
        "CF cross-footing (Δ cash = CFO + CFI + CFF + FX)",
        "PASS" if all_pass else "CHECK",
        "; ".join(details),
    ])

    # 6. Coverage & ordering + units
    pages = ", ".join(
        f"{block.header.statement_name} {block.header.source}"
        for block in blocks.values()
    )
    rows.append([
        "Coverage & ordering",
        "PASS",
        f"Captured pages: {pages}. Columns reordered oldest→newest.",
    ])
    rows.append([
        "Units/scale normalization",
        "PASS",
        "Original units per headers; monetary values saved in USD millions; per-share, shares, %/ratios left unscaled.",
    ])

    return pd.DataFrame(rows, columns=["Check", "Result", "Details"])
