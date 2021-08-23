import importlib
import os
import sys

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
VARIABLES_IMPORT = "variables"
OUTPUT_DIR = "output"
VARIABLES_TXT = "variables.txt"
ROOT_DIR_TXT = "root.txt"

if __name__ == "__main__":
    def get_line(args):
        try:
            return f"{' '.join([str(arg) for arg in args])}\n"
        except TypeError:
            # args is not iterable
            return f"{args}\n"

    # get the directory containing the simulations
    root_dir = sys.argv[1]
    if not os.path.exists(root_dir):
        raise ValueError(f"Directory {root_dir} does not exist.")

    # write variables to file
    variables = importlib.import_module(f".{VARIABLES_IMPORT}", package=root_dir)
    with open(VARIABLES_TXT, "w") as f:
        f.writelines([get_line(args) for args in variables.variables])

    # store root directory
    with open(ROOT_DIR_TXT, "w") as f:
        f.write(root_dir)

    # create directory for cluster output
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    