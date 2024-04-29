from project.labs.lab02 import lab2
from project.labs.lab03 import lab3
from project.labs.lab04 import lab4
from project.labs.lab05 import lab5

DATA = "data/trainData.txt"
CONF = {
    "lab2": False,
    "lab3": False,
    "lab4": False,
    "lab5": True,
}


def main():
    # TODO: Enable this once python 3.12 support is added
    # os.environ["MPLBACKEND"] = "module://mplcairo.base"

    if CONF["lab2"]:
        lab2(DATA)

    if CONF["lab3"]:
        lab3(DATA)

    if CONF["lab4"]:
        lab4(DATA)

    if CONF["lab5"]:
        lab5(DATA)


if __name__ == "__main__":
    main()
