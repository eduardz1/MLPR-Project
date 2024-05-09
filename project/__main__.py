import argparse

import typst
from rich.console import Console
from rich.markdown import Markdown
from rich.status import Status

from project.labs.lab02 import lab2
from project.labs.lab03 import lab3
from project.labs.lab04 import lab4
from project.labs.lab05 import lab5

TYPST_PATH = "report/report.typ"
DATA = "data/trainData.txt"

conf = {
    "lab2": False,
    "lab3": False,
    "lab4": False,
    "lab5": False,
    "compile_pdf": False,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the code for the project, separately for each labs. Optionally compile the report."
    )
    parser.add_argument(
        "--lab2", action="store_true", help="run project part for lab 2"
    )
    parser.add_argument(
        "--lab3", action="store_true", help="run project part for lab 3"
    )
    parser.add_argument(
        "--lab4", action="store_true", help="run project part for lab 4"
    )
    parser.add_argument(
        "--lab5", action="store_true", help="run project part for lab 5"
    )
    parser.add_argument(
        "-c", "--compile_pdf", action="store_true", help="compile the report"
    )

    args = parser.parse_args()

    if args.lab2:
        conf["lab2"] = True
    if args.lab3:
        conf["lab3"] = True
    if args.lab4:
        conf["lab4"] = True
    if args.lab5:
        conf["lab5"] = True
    if args.compile_pdf:
        conf["compile_pdf"] = True


def main():
    console = Console()

    # TODO: Enable this once python 3.12 support is added
    # https://github.com/matplotlib/mplcairo/issues/51
    # os.environ["MPLBACKEND"] = "module://mplcairo.base"

    parse_args()

    if conf["lab2"]:
        console.print(Markdown("# Lab 2"), new_line_start=True)
        lab2(DATA)

    if conf["lab3"]:
        console.print(Markdown("# Lab 3"), new_line_start=True)
        lab3(DATA)

    if conf["lab4"]:
        console.print(Markdown("# Lab 4"), new_line_start=True)

        lab4(DATA)
    if conf["lab5"]:
        console.print(Markdown("# Lab 5"), new_line_start=True)
        lab5(DATA)

    if conf["compile_pdf"]:
        status = Status("Compiling the report...")
        status.start()
        typst.compile(TYPST_PATH, output=TYPST_PATH.replace(".typ", ".pdf"))
        status.stop()


if __name__ == "__main__":
    main()
