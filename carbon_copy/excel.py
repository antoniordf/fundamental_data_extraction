from __future__ import annotations

from pathlib import Path

import pandas as pd

from selector.models import StatementType

from .models import CarbonCopyResult


WIDE_SHEETS = {
    StatementType.INCOME: "WIDE_Income",
    StatementType.BALANCE: "WIDE_BalanceSheet",
    StatementType.CASH: "WIDE_CashFlows",
}


def write_excel_workbook(result: CarbonCopyResult, output_path: Path) -> None:
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        # Pre-create worksheets
        for sheet_name in [*WIDE_SHEETS.values(), "LONG_All", "Certification"]:
            writer.book.add_worksheet(sheet_name)

        def write_block(statement: StatementType) -> None:
            block = result.blocks[statement]
            sheet_name = WIDE_SHEETS[statement]
            worksheet = writer.sheets[sheet_name]
            header_dict = block.header.as_dict()
            row_idx = 0
            for key, value in header_dict.items():
                worksheet.write(row_idx, 0, key)
                worksheet.write(row_idx, 1, value)
                row_idx += 1
            block.wide_table.to_excel(
                writer,
                sheet_name=sheet_name,
                startrow=row_idx + 1,
                index=False,
            )

        for statement in [StatementType.INCOME, StatementType.BALANCE, StatementType.CASH]:
            write_block(statement)

        result.long_table.to_excel(writer, sheet_name="LONG_All", index=False)
        result.certification.to_excel(writer, sheet_name="Certification", index=False)
