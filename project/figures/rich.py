from typing import Any

import numpy as np
import numpy.typing as npt
from rich.console import Console
from rich.table import Table, box


def table(
    console: Console,
    title: str,
    dictionary: dict[str, list | Any],
    llr: npt.ArrayLike | None = None,
    render: bool = True,
):
    table = Table(title=title, box=box.ROUNDED)

    keys = np.array(list(dictionary.keys()))

    # Make sure all values are arrays
    values = [np.atleast_1d(value) for value in dictionary.values()]

    for key in keys:
        table.add_column(key, justify="center")

    for i in range(len(values[0])):
        row = [
            (
                f"{value[i]:.4f}"
                if isinstance(value[i], float)
                else (
                    f"{*value[i],}"  # noqa: E231
                    if isinstance(value[i], list)
                    else f"{value[i]}"
                )
            )
            for value in values
        ]
        table.add_row(*row)

    if render:
        console.print(table, new_line_start=True)

    if llr is not None:
        console.print(f"Log-likelihood ratio: {llr}")

    return table
