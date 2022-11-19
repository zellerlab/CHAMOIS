import argparse
import datetime
import json
import os
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
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

# Create a session and spoof the user agent so that JGI lets us use programmatic access
session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64; rv:78.0) Gecko/20100101 Firefox/78.0"

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
    # load genomes
    console.print(f"[bold green]{'Saving':>12}[/] ABC genomes to 'img_abc.json'")
    with open(args.input, "rb") as f:
        records = json.load(f)

    # extract all clusters
    experimental_clusters = []
    for record in progress.track(records, description="Browsing genomes...", total=len(records)):

        progress.console.print(f"[bold green]{'Getting':>12}[/] clusters from [purple]{record['GenomeName']}[/]")
        params = {"section":"BiosyntheticDetail", "page":"biosynthetic_clusters", "taxon_oid":record["GenomeID"]}
        with session.get(f"{CGI_URL}/abc-public/main.cgi", params=params) as res:
            progress.console.print(f"[bold blue]{'Responded':>12}[/]", "with HTTP code", res.status_code)
            progress.console.print(f"[bold green]{'Extracting':>12}[/]", "CGI data source from JavaScript code")
            RX_DATA_SOURCE = re.compile(r'YAHOO\.util\.DataSource\("([^(]*)"\)')
            match = RX_DATA_SOURCE.search(res.text)
        if match is None:
            progress.console.print(f"[bold red]{'Failed':>12}[/] to locate the JSON data source")
            exit(1)
        parsed_source = urllib.parse.urlparse(match.group(1))
        parsed_params = urllib.parse.parse_qs(parsed_source.query)

        params = {
            "sid": parsed_params["sid"],
            "results": 100,
            "startIndex": 0,
            "sort": "ClusterID",
            "dir": "asc",
            "c": "",
            "f": "",
            "t": "",
            "callid": int(datetime.datetime.now().timestamp()) * 1000,
            "cached_session": parsed_params["cached_session"],
        }
        with session.get(f"{CGI_URL}/abc-public/{parsed_source.path}", params=params) as res:
            progress.console.print(f"[bold blue]{'Responded':>12}[/]", "with HTTP code", res.status_code)
            clusters = res.json()

        if len(clusters["records"]) < clusters["totalRecords"]:
            progress.console.print(f"[bold red]{'Failed':>12}[/] to retrieve all clusters")
            exit(1)

        for cluster_record in clusters["records"]:
            if cluster_record["Method"] != "Experimental":
                progress.console.print(f"[bold blue]{'Skipping':>12}[/] non-experimental cluster [purple]{cluster_record['ClusterID']}[/]")
                continue
            if cluster_record.get("MIBiGLinks", ""):
                progress.console.print(f"[bold blue]{'Skipping':>12}[/] cluster [purple]{cluster_record['ClusterID']}[/] already in MIBiG as [purple]{cluster_record['MIBiGLinks']}[/]")
                continue
            if "SecondaryMetaboliteDisp" not in cluster_record:
                progress.console.print(f"[bold blue]{'Skipping':>12}[/] cluster [purple]{cluster_record['ClusterID']}[/] without known secondary metabolites")
                continue

            compounds = BeautifulSoup(cluster_record["SecondaryMetaboliteDisp"], "html.parser")
            compound_links = {link.text.strip():link.next_sibling.strip() for link in compounds.find_all("a")}

            cluster_record["GenomeID"] = record["GenomeID"]
            experimental_clusters.append(cluster_record)

            progress.console.print(
                f"[bold green]{'Storing':>12}[/] experimental cluster [purple]{cluster_record['ClusterID']}[/]", 
                "of unknown type" if not cluster_record["BGCType"] else f"of type [purple]{cluster_record['BGCType']}[/]",
                "producing", ", ".join(set(compound_links.values())) if compound_links else "unknown compound",
            )


# Save genome results
console.print(f"[bold green]{'Saving':>12}[/] ABC clusters to {args.output!r}")
os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump(experimental_clusters, f, indent=4, sort_keys=True)
