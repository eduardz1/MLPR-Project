import argparse
import contextlib
import os

import typst
from rich.console import Console
from rich.status import Status

from project.labs.lab02 import lab02
from project.labs.lab03 import lab03
from project.labs.lab04 import lab04
from project.labs.lab05 import lab05
from project.labs.lab07 import lab07
from project.labs.lab08 import lab08
from project.labs.lab09 import lab09

TYPST_PATH = "report/report.typ"
DATA = "data/trainData.txt"

conf = {
    "lab02": False,
    "lab03": False,
    "lab04": False,
    "lab05": False,
    "lab07": False,
    "lab08": False,
    "lab09": False,
    "compile_pdf": False,
    "quiet": False,
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
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress additional information during code execution",
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
        choices=[2, 3, 4, 5, 7, 8, 9],
        type=int,
        nargs="+",
        help="run specific project parts by specifying one of more associated lab numbers",
    )

    args = parser.parse_args()

    if args.all:
        conf["lab02"] = True
        conf["lab03"] = True
        conf["lab04"] = True
        conf["lab05"] = True
        conf["lab07"] = True
        conf["lab08"] = True
        conf["lab09"] = True
    else:
        for lab in args.labs:
            if lab == 2:
                conf["lab02"] = True
            elif lab == 3:
                conf["lab03"] = True
            elif lab == 4:
                conf["lab04"] = True
            elif lab == 5:
                conf["lab05"] = True
            elif lab == 7:
                conf["lab07"] = True
            elif lab == 8:
                conf["lab08"] = True
            elif lab == 9:
                conf["lab09"] = True
    if args.compile_pdf:
        conf["compile_pdf"] = True
    if args.quiet:
        conf["quiet"] = True


def main():
    console = Console()

    # TODO: Enable this once python 3.12 support is added
    # https://github.com/matplotlib/mplcairo/issues/51
    # os.environ["MPLBACKEND"] = "module://mplcairo.base"
    # os.environ["MPLBACKEND"] = "Agg"

    parse_args()

    def run_labs():
        if conf["lab02"]:
            console.log("[bold red]Lab 2 - Analyzing the features [/bold red]")
            lab02(DATA)

        if conf["lab03"]:
            console.log("[bold red]Lab 3 - PCA & LDA [/bold red]")
            lab03(DATA)

        if conf["lab04"]:
            console.log(
                "[bold red]Lab 4 - Probability densities and ML estimates [/bold red]"
            )
            lab04(DATA)

        if conf["lab05"]:
            console.log(
                "[bold red]Lab 5 - Generative models for classification [/bold red]"
            )
            lab05(DATA)

        if conf["lab07"]:
            console.log(
                "[bold red]Lab 7 - Performance analysis of the MVG classifier [/bold red]"
            )
            lab07(DATA)

        if conf["lab08"]:
            console.log(
                "[bold red]Lab 8 - Performance analysis of the Binary Logistic Regression classifier [/bold red]"
            )
            lab08(DATA)

        if conf["lab09"]:
            console.log(
                "[bold red]Lab 9 - Support Vector Machines classification [/bold red]"
            )
            lab09(DATA)

        if conf["compile_pdf"]:
            status = Status("Compiling the report...")
            status.start()
            typst.compile(TYPST_PATH, output=TYPST_PATH.replace(".typ", ".pdf"))
            status.stop()

    if conf["quiet"]:
        # Suppress output by redirecting stdout to /dev/null
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            run_labs()
    else:
        run_labs()


if __name__ == "__main__":
    main()
