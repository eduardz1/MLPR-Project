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
from project.labs.lab10 import lab10
from project.labs.lab11 import lab11

TYPST_PATH = "report/report.typ"
DATA = "data/trainData.txt"

lab_config = {
    "lab02": {
        "enabled": False,
        "function": lab02,
        "title": "Analyzing the features",
    },
    "lab03": {
        "enabled": False,
        "function": lab03,
        "title": "PCA & LDA",
    },
    "lab04": {
        "enabled": False,
        "function": lab04,
        "title": "Probability densities and ML estimates",
    },
    "lab05": {
        "enabled": False,
        "function": lab05,
        "title": "Generative models for classification",
    },
    "lab07": {
        "enabled": False,
        "function": lab07,
        "title": "Performance analysis of the MVG classifier",
    },
    "lab08": {
        "enabled": False,
        "function": lab08,
        "title": "Performance analysis of the Binary Logistic Regression classifier",
    },
    "lab09": {
        "enabled": False,
        "function": lab09,
        "title": "Support Vector Machines classification",
    },
    "lab10": {
        "enabled": False,
        "function": lab10,
        "title": "Gaussian Mixture Models",
    },
    "lab11": {
        "enabled": False,
        "function": lab11,
        "title": "Calibration and Fusion",
    },
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
        choices=[2, 3, 4, 5, 7, 8, 9, 10, 11],
        type=int,
        nargs="+",
        help="run specific project parts by specifying one of more associated lab numbers",
    )

    args = parser.parse_args()

    if args.all:
        for lab_key in lab_config.keys():
            lab_config[lab_key]["enabled"] = True
    else:
        for lab in args.labs:
            # Converts 2 to "lab02", 10 to "lab10", etc.
            lab_key = f"lab{str(lab).zfill(2)}"
            if lab_key in lab_config:
                lab_config[lab_key]["enabled"] = True

    return args


def main():
    console = Console()

    # TODO: Enable this once python 3.12 support is added
    # https://github.com/matplotlib/mplcairo/issues/51
    # os.environ["MPLBACKEND"] = "module://mplcairo.base"
    os.environ["MPLBACKEND"] = "Agg"

    args = parse_args()

    def run_labs():
        for lab_id, lab_info in lab_config.items():
            if lab_info["enabled"]:
                console.log(f"{lab_id} - [bold red]{lab_info['title']} [/bold red]")
                lab_info["function"](DATA)

        if args.compile_pdf:
            status = Status("Compiling the report...")
            status.start()
            typst.compile(TYPST_PATH, output=TYPST_PATH.replace(".typ", ".pdf"))
            status.stop()

    if args.quiet:
        # Suppress output by redirecting stdout to /dev/null
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            run_labs()
    else:
        run_labs()


if __name__ == "__main__":
    main()
