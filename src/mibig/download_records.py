import argparse
import urllib.request
import os
import io
import tarfile

import rich.progress
import pandas
import Bio.SeqIO


parser = argparse.ArgumentParser()
parser.add_argument("--blocklist")
parser.add_argument("--mibig-version", default="3.1")
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

url = f"https://dl.secondarymetabolites.org/mibig/mibig_gbk_{args.mibig_version}.tar.gz"


def extract_records(response):
    total = int(response.headers["Content-Length"])
    with progress.wrap_file(response, total=total, description="Downloading...") as f:
        with tarfile.open(fileobj=f, mode="r|gz") as tar:
            for entry in iter(tar.next, None):
                if entry.name.endswith(".gbk"):
                    with tar.extractfile(entry) as f:
                        data = io.StringIO(f.read().decode())
                        yield Bio.SeqIO.read(data, "genbank")


def get_cds(record, **kwargs):
    for k, v in kwargs.items():
        break
    return next(
        f
        for f in record.features
        if f.type == "CDS"
        and f.qualifiers.get(k, [""])[0] == v
    )


with rich.progress.Progress() as progress:

    # load blocklist if any
    if args.blocklist is not None:
        table = pandas.read_table(args.blocklist)
        blocklist = set(table.bgc_id.unique())
    else:
        blocklist = set()

    # download MIBIG 3 records
    with urllib.request.urlopen(url) as response:

        records = []
        for record in extract_records(response):
            # ignore BGCs in blocklist
            if record.id in blocklist:
                continue

            # BGC0000017 has unneeded downstream genes, only keep until `anaH`
            # (genes are characterized in doi:10.1016/j.toxicon.2014.07.016)
            if record.id == "BGC0000017":
                start = 0
                end = get_cds(record, gene="anaH").location.end

            # BGC0000122 has additional genes, the core BGC is only composed
            # of phn1-phn2 (see doi:10.1002/cbic.201300676)
            elif record.id == "BGC0000122":
                start = get_cds(record, gene="phn2").location.start
                end = get_cds(record, gene="phn1").location.end

            # BGC0000938 has many uneeded genes, some of which have been shown
            # to be unneeded for biosynthesis through deletion analysis, and
            # the authors only consider fom3-fomC to be the core fosfomycin BGC
            # (see doi:10.1016/j.chembiol.2006.09.007)
            elif record.id == "BGC0000938":
                start = get_cds(record, gene="fom3").location.start
                end = get_cds(record, gene="fomC").location.end

            # BGC0001367 has unneeded downstream genes, the authors only
            # consider the two core genes `hla1` and `hla2` to be part of
            # the BGC and sufficient for synthesis of the final compound
            # (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6274090/)
            elif record.id == "BGC0001367":
                start = get_cds(record, locus_tag="Hoch_0798").location.start
                end = get_cds(record, locus_tag="Hoch_0799").location.end

            # Heterologous expression of BGC0001917 shows that only `stmA` to `stmI`
            # are required for synthesis (see doi:10.1039/c8ob02846j, Fig.3)
            elif record.id == "BGC00001917":
                start = get_cds(record, gene="stmA").location.start
                end = get_cds(record, gene="stmI").location.end

            # MIBiG entry of BGC0001202 contains unrelated upstream and
            # downstream genes
            elif record.id == "BGC0001202":
                start = get_cds(record, protein_id="AKA59430.1").location.start
                end = get_cds(record, protein_id=" AKA59442.1").location.end

            # clamp the BGC boundaries to the left- and rightmost genes
            else:
                start = min( f.location.start for f in record.features if f.type == "CDS" )
                end = max( f.location.end for f in record.features if f.type == "CDS" )

            # copy
            assert start < end, f"{record.id}: {start} < {end}"
            bgc_record = record[start:end]
            bgc_record.annotations = record.annotations.copy()
            bgc_record.id = record.id
            bgc_record.name = record.name
            bgc_record.description = record.description
            records.append(bgc_record)

    # sort records by MIBiG accession
    records.sort(key=lambda record: record.id)

    # save records
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as dst:
        Bio.SeqIO.write(records, dst, "genbank")
