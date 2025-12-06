import argparse
import urllib.request
import tarfile
import json
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--version", default="1.18")
parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
args = parser.parse_args()

entries = []
url = f"https://github.com/mite-standard/mite_data/archive/{args.version}.tar.gz"
with urllib.request.urlopen(url) as res:
    with tarfile.open(fileobj=res, mode="r|gz") as tar:
        for member in iter(tar.next, None):
            name = pathlib.Path(member.name)
            if name.name.startswith("MITE") and name.name.endswith(".json"):
                with tar.extractfile(member) as f:
                    data = json.load(f)
                    entries.append(data)

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w") as dst:
    json.dump(entries, dst, sort_keys=True, indent=4)
