from __future__ import annotations

from pathlib import Path

import pandas as pd

from selector.models import StatementType

from .certification import build_certification
from .constants import DEMO_AUDITED_LABEL, SAVED_UNITS_LABEL
from .models import CarbonCopyResult, HeaderInfo, StatementBlock
from .utils import assemble_long


def _frame(columns, rows):
    df = pd.DataFrame(rows, columns=columns)
    return df


def build_demo_result(output_path: Path | None = None) -> CarbonCopyResult:
    income_columns = ["Order Index", "Level", "Line Item (as printed)", "2023", "2024", "2025"]
    income_rows = [
        (1, 0, "Net revenue", 7426, 7562, 7463),
        (2, 0, "Cost of revenue", -1792, -1710, -1543),
        (3, 0, "Gross profit", 5634, 5852, 5920),
        (4, 0, "Operating expenses:", None, None, None),
        (5, 1, "Research and development", -2328, -2420, -2569),
        (6, 1, "Marketing and sales", -978, -1019, -962),
        (7, 1, "General and administrative", -727, -691, -745),
        (8, 1, "Amortization and impairment of intangibles", -158, -142, -67),
        (9, 1, "Restructuring (See Note 8)", -111, -62, -57),
        (10, 0, "Total operating expenses", -4302, -4334, -4400),
        (11, 0, "Operating income", 1332, 1518, 1520),
        (12, 0, "Interest and other income (expense), net", -6, 71, 85),
        (13, 0, "Income before provision for income taxes", 1326, 1589, 1605),
        (14, 0, "Provision for income taxes", -524, -316, -484),
        (15, 0, "Net income", 802, 1273, 1121),
        (16, 0, "Earnings per share:", None, None, None),
        (17, 1, "Basic", 2.90, 4.71, 4.28),
        (18, 1, "Diluted", 2.88, 4.68, 4.25),
        (19, 0, "Number of shares used in computation:", None, None, None),
        (20, 1, "Basic", 277, 270, 262),
        (21, 1, "Diluted", 278, 272, 264),
    ]
    income_df = _frame(income_columns, income_rows)
    income_header = HeaderInfo(
        statement_name="CONSOLIDATED STATEMENTS OF OPERATIONS",
        original_units="In millions, except per share data",
        saved_units=SAVED_UNITS_LABEL,
        units_assumption=None,
        audited_status=DEMO_AUDITED_LABEL,
        source="p.35",
    )
    income_block = StatementBlock(StatementType.INCOME, income_header, income_df)

    balance_columns = [
        "Order Index",
        "Level",
        "Line Item (as printed)",
        "March 31, 2024",
        "March 31, 2025",
    ]
    balance_rows = [
        (1, 0, "ASSETS", None, None),
        (2, 1, "Current assets:", None, None),
        (3, 2, "Cash and cash equivalents", 2900, 2136),
        (4, 2, "Short-term investments", 362, 112),
        (5, 2, "Receivables, net", 565, 679),
        (6, 2, "Other current assets", 420, 349),
        (7, 1, "Total current assets", 4247, 3276),
        (8, 1, "Property and equipment, net", 578, 586),
        (9, 1, "Goodwill", 5379, 5376),
        (10, 1, "Acquisition-related intangibles, net", 400, 293),
        (11, 1, "Deferred income taxes, net", 2380, 2420),
        (12, 1, "Other assets", 436, 417),
        (13, 0, "TOTAL ASSETS", 13420, 12368),
        (14, 0, "LIABILITIES AND STOCKHOLDERS’ EQUITY", None, None),
        (15, 1, "Current liabilities:", None, None),
        (16, 2, "Accounts payable, accrued, and other current liabilities", 1276, 1359),
        (17, 2, "Deferred net revenue (online-enabled games)", 1814, 1700),
        (18, 2, "Senior notes, current, net", None, 400),
        (19, 1, "Total current liabilities", 3090, 3459),
        (20, 1, "Senior notes, net", 1882, 1484),
        (21, 1, "Income tax obligations", 497, 594),
        (22, 1, "Other liabilities", 438, 445),
        (23, 0, "Total liabilities", 5907, 5982),
        (24, 0, "Commitments and contingencies (See Note 14)", None, None),
        (25, 1, "Stockholders’ equity:", None, None),
        (26, 2, "Preferred stock, $0.01 par value. 10 shares authorized", None, None),
        (27, 2, "Common stock, $0.01 par value. 1,000 shares authorized; 252 and 266 shares issued and outstanding, respectively", 3, 3),
        (28, 2, "Additional paid-in capital", None, None),
        (29, 2, "Retained earnings", 7582, 6470),
        (30, 2, "Accumulated other comprehensive income (loss)", -72, -87),
        (31, 1, "Total stockholders’ equity", 7513, 6386),
        (32, 0, "TOTAL LIABILITIES AND STOCKHOLDERS’ EQUITY", 13420, 12368),
    ]
    balance_df = _frame(balance_columns, balance_rows)
    balance_header = HeaderInfo(
        statement_name="CONSOLIDATED BALANCE SHEETS",
        original_units="In millions, except par value data",
        saved_units=SAVED_UNITS_LABEL,
        units_assumption=None,
        audited_status=DEMO_AUDITED_LABEL,
        source="p.34",
    )
    balance_block = StatementBlock(StatementType.BALANCE, balance_header, balance_df)

    cash_columns = ["Order Index", "Level", "Line Item (as printed)", "2023", "2024", "2025"]
    cash_rows = [
        (1, 0, "OPERATING ACTIVITIES", None, None, None),
        (2, 1, "Net income", 802, 1273, 1121),
        (3, 1, "Adjustments to reconcile net income to net cash provided by operating activities:", None, None, None),
        (4, 2, "Depreciation, amortization, accretion and impairment", 536, 404, 356),
        (5, 2, "Stock-based compensation", 548, 584, 642),
        (6, 1, "Change in assets and liabilities:", None, None, None),
        (7, 2, "Receivables, net", -34, 119, -115),
        (8, 2, "Other assets", -103, 148, 40),
        (9, 2, "Accounts payable, accrued, and other liabilities", 144, -208, 190),
        (10, 2, "Deferred income taxes, net", -221, 82, -41),
        (11, 2, "Deferred net revenue (online-enabled games)", -122, -87, -114),
        (12, 1, "Net cash provided by operating activities", 1550, 2315, 2079),
        (13, 0, "INVESTING ACTIVITIES", None, None, None),
        (14, 1, "Capital expenditures", -207, -199, -221),
        (15, 1, "Proceeds from maturities and sales of short-term investments", 395, 632, 695),
        (16, 1, "Purchase of short-term investments", -405, -640, -437),
        (17, 1, "Net cash provided by (used in) investing activities", -217, -207, 37),
        (18, 0, "FINANCING ACTIVITIES", None, None, None),
        (19, 1, "Proceeds from issuance of common stock", 80, 77, 78),
        (20, 1, "Cash dividends paid", -210, -205, -199),
        (21, 1, "Cash paid to taxing authorities for shares withheld from employees", -175, -196, -234),
        (22, 1, "Common stock repurchases and excise taxes paid", -1295, -1300, -2508),
        (23, 1, "Net cash used in financing activities", -1600, -1624, -2863),
        (24, 0, "Effect of foreign exchange on cash and cash equivalents", -41, -8, -17),
        (25, 0, "Increase (decrease) in cash and cash equivalents", -308, 476, -764),
        (26, 0, "Beginning cash and cash equivalents", 2732, 2424, 2900),
        (27, 0, "Ending cash and cash equivalents", 2424, 2900, 2136),
        (28, 0, "Supplemental cash flow information:", None, None, None),
        (29, 1, "Cash paid during the year for income taxes, net", -583, -300, -404),
        (30, 1, "Cash paid during the year for interest", -56, -56, -56),
        (31, 0, "Non-cash investing activities:", None, None, None),
        (32, 1, "Change in accrued capital expenditures", -3, 25, None),
    ]
    cash_df = _frame(cash_columns, cash_rows)
    cash_header = HeaderInfo(
        statement_name="CONSOLIDATED STATEMENTS OF CASH FLOWS",
        original_units="In millions",
        saved_units=SAVED_UNITS_LABEL,
        units_assumption=None,
        audited_status=DEMO_AUDITED_LABEL,
        source="p.38",
    )
    cash_block = StatementBlock(StatementType.CASH, cash_header, cash_df)

    blocks = {
        StatementType.INCOME: income_block,
        StatementType.BALANCE: balance_block,
        StatementType.CASH: cash_block,
    }

    long = pd.concat(
        [
            assemble_long(StatementType.INCOME, income_header.as_dict(), income_df, income_header.source),
            assemble_long(StatementType.BALANCE, balance_header.as_dict(), balance_df, balance_header.source),
            assemble_long(StatementType.CASH, cash_header.as_dict(), cash_df, cash_header.source),
        ],
        ignore_index=True,
    )

    certification = build_certification(blocks)
    return CarbonCopyResult(Path("demo"), blocks, long, certification)
