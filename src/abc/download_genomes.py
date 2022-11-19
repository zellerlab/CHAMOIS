import argparse
import datetime
import json
import re
import urllib.parse
from pprint import pprint

import requests
import rich.console
import rich.progress
from bs4 import BeautifulSoup

CONSOLE = console = rich.console.Console()
CGI_URL = "https://img.jgi.doe.gov/cgi-bin"

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

# Create a session and spoof the user agent so that JGI lets us use programmatic access
session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64; rv:78.0) Gecko/20100101 Firefox/78.0"

# Get the basic search page
console.print(f"[bold green]{'Accessing':>12}[/]", "IMG/ABC public portal")
params = {"section": "BiosyntheticStats", "page": "experimentalBcGenomes"}
with session.get(f"{CGI_URL}/abc-public/main.cgi", params=params) as res:
    console.print(f"[bold blue]{'Responded':>12}[/]", "with HTTP code", res.status_code)
    console.print(f"[bold green]{'Extracting':>12}[/]", "CGI data source from JavaScript code")
    RX_DATA_SOURCE = re.compile(r'YAHOO\.util\.DataSource\("([^(]*)"\)')
    match = RX_DATA_SOURCE.search(res.text)
if match is None:
    console.print(f"[bold red]{'Failed':>12}[/] to locate the JSON data source")
    exit(1)
parsed_source = urllib.parse.urlparse(match.group(1))
parsed_params = urllib.parse.parse_qs(parsed_source.query)

# Get all genomes 
with rich.progress.Progress(
     rich.progress.SpinnerColumn(finished_text="[green]:heavy_check_mark:[/]"),
     "[progress.description]{task.description}",
     rich.progress.BarColumn(bar_width=60),
     "[progress.completed]{task.completed}/{task.total}",
     "[progress.percentage]{task.percentage:>3.0f}%",
     rich.progress.TimeElapsedColumn(),
     rich.progress.TimeRemainingColumn(),
     console=console,
     transient=True,
) as progress:
    # Get all genomes with experimentally validated BGCs in them
    console.print(f"[bold green]{'Searching':>12}[/] for experimentally validated BGCs")
    records = []
    total_records = float("+inf")  # we only know this after the first response
    task = progress.add_task(total=None, description="Searching...")
    while len(records) < total_records:
        params = {
            "sid": parsed_params["sid"],
            "results": 100,
            "startIndex": len(records),
            "sort": "Domain",
            "dir": "asc",
            "c": "",
            "f": "",
            "t": "",
            "callid": int(datetime.datetime.now().timestamp()) * 1000,
            "cached_session": parsed_params["cached_session"],
        }
        with session.get(f"{CGI_URL}/abc-public/{parsed_source.path}", params=params) as res:
            console.print(f"[bold blue]{'Responded':>12}[/]", "with HTTP code", res.status_code)
            data = res.json()
        records.extend(data["records"])
        total_records = data["totalRecords"]
        progress.update(task_id=task, total=total_records, advance=len(data["records"]))

    # Save genome results
    console.print(f"[bold green]{'Saving':>12}[/] ABC genomes to 'img_abc.json'")
    with open(args.output, "w") as f:
        json.dump(records, f, indent=4, sort_keys=True)
