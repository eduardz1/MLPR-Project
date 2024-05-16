from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from rich.console import Console
from rich.table import Table, box


def table(
    console: Console,
    title: str,
    dictionary: dict[str, list],
    llr: Optional[ArrayLike] = None,
):
    table = Table(title=title, box=box.ROUNDED)

    keys = np.array(list(dictionary.keys()))
    values = np.array(list(dictionary.values()))

    for key in keys:
        table.add_column(key, justify="center")

    for i in range(len(values[0])):
        row = [f"{value[i]}" for value in values]
        table.add_row(*row)

    console.print(table, new_line_start=True)

    if llr is not None:
        console.print(f"Log-likelihood ratio: {llr}")
