import sys
from pathlib import Path
parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent))

from specparser import specparser
from specparser import expander

import argparse

p = argparse.ArgumentParser("spec",description=("exstract spec"))

p.add_argument("spec",help="spec string.")
p.add_argument(
    "--name",
    type=str,
    default="slot",
    help="item name.",
)
p.add_argument(
    "--ndx",
    type=int,
    default=0,
    help="item index.",
)

args = p.parse_args()

spec_dict = specparser.split_chain(args.spec)

print(spec_dict[args.name][args.ndx])