import argparse
import os

import typst
from rich.console import Console
from rich.status import Status

from project.labs.lab02 import lab2
from project.labs.lab03 import lab3
from project.labs.lab04 import lab4
from project.labs.lab05 import lab5
from project.labs.lab07 import lab7

TYPST_PATH = "report/report.typ"
DATA = "data/trainData.txt"

conf = {
    "lab2": False,
    "lab3": False,
    "lab4": False,
    "lab5": False,
    "lab7": False,
    "compile_pdf": False,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the code for the project, separately for each labs. Optionally compile the report."
    )
    parser.add_argument(
        "-c",
        "--compile_pdf",
        action="store_true",
        help="compile the report in pdf format",
    )
    group = parser.add_argument_group(
        "labs", "Choose which part of the project, ordered by labs, to run"
    )

    exclusive_group = group.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="run all project parts (does not compile the report)",
    )
    exclusive_group.add_argument(
        "-l",
        "--labs",
        choices=[2, 3, 4, 5, 7],
        type=int,
        nargs="+",
        help="run specific project parts by specifying one of more associated lab numbers",
    )

    args = parser.parse_args()

    if args.all:
        conf["lab2"] = True
        conf["lab3"] = True
        conf["lab4"] = True
        conf["lab5"] = True
        conf["lab7"] = True
    else:
        for lab in args.labs:
            if lab == 2:
                conf["lab2"] = True
            elif lab == 3:
                conf["lab3"] = True
            elif lab == 4:
                conf["lab4"] = True
            elif lab == 5:
                conf["lab5"] = True
            elif lab == 7:
                conf["lab7"] = True
    if args.compile_pdf:
        conf["compile_pdf"] = True


def main():
    console = Console()

    # TODO: Enable this once python 3.12 support is added
    # https://github.com/matplotlib/mplcairo/issues/51
    # os.environ["MPLBACKEND"] = "module://mplcairo.base"
    os.environ["MPLBACKEND"] = "Agg"

    parse_args()

    if conf["lab2"]:
        console.log("[bold red]Lab 2 - Analyzing the features [/bold red]")
        lab2(DATA)

    if conf["lab3"]:
        console.log("[bold red]Lab 3 - PCA & LDA [/bold red]")
        lab3(DATA)

    if conf["lab4"]:
        console.log(
            "[bold red]Lab 4 - Probability densities and ML estimates [/bold red]"
        )
        lab4(DATA)

    if conf["lab5"]:
        console.log(
            "[bold red]Lab 5 - Generative models for classification [/bold red]"
        )
        lab5(DATA)

    if conf["lab7"]:
        console.log(
            "[bold red]Lab 7 - Performance analysis of the MVG classifier [/bold red]"
        )
        lab7(DATA)

    if conf["compile_pdf"]:
        status = Status("Compiling the report...")
        status.start()
        typst.compile(TYPST_PATH, output=TYPST_PATH.replace(".typ", ".pdf"))
        status.stop()


if __name__ == "__main__":
    main()
