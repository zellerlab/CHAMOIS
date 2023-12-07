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

            # BGC0000078 includes several ORFs marked outside of the gene
            # cluster by the authors
            # (see https://pubmed.ncbi.nlm.nih.gov/23921821/, Table 1)
            elif record.id == "BGC0000078":
                start = get_cds(record, gene="idnA").location.start
                end = get_cds(record, gene="idnM5").location.end

            # lasalocid BGCs contains unrelated flanking genes; the reference
            # identifies only the core 16 genes, `las1` to `lasB`, as the BGC
            # (see doi:10.1002/cbic.200800585)
            elif record.id == "BGC0000086":
                start = get_cds(record, gene="lsd1").location.start
                end = get_cds(record, gene="lsd19").location.end
            elif record.id == "BGC0000087":
                start = get_cds(record, gene="las1").location.start
                end = get_cds(record, gene="lasB").location.end

            # BGC0000108 has a couple of unrelated upstream genes
            # (see https://pubmed.ncbi.nlm.nih.gov/21330439/, Figure 1)
            elif record.id == "BGC0000108":
                start = get_cds(record, gene="scnRII").location.start
                end = get_cds(record, gene="scnD").location.end

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

            # BGC0000966 has some flanking genes in the cluster that are not
            # counted as part of the BGC by the authors
            # (see https://pubmed.ncbi.nlm.nih.gov/22591508/, Figure 1)
            elif record.id == "BGC0000966":
                start = get_cds(record, protein_id="AFD30965.1").location.start
                end = get_cds(record, protein_id="AFD30946.1").location.end

            # MIBIG entry of BGC0001196 contains ORF-2 to ORF+3
            elif record.id == "BGC0001196":
                start = get_cds(record, protein_id="AJW76703.1").location.start
                end = get_cds(record, protein_id="AJW76719.1").location.end

            # MIBiG entry of BGC0001202 contains unrelated upstream and
            # downstream genes
            elif record.id == "BGC0001202":
                start = get_cds(record, protein_id="AKA59430.1").location.start
                end = get_cds(record, protein_id="AKA59442.1").location.end

            # BGC0001367 has unneeded downstream genes, the authors only
            # consider the two core genes `hla1` and `hla2` to be part of
            # the BGC and sufficient for synthesis of the final compound
            # (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6274090/)
            elif record.id == "BGC0001367":
                start = get_cds(record, locus_tag="Hoch_0798").location.start
                end = get_cds(record, locus_tag="Hoch_0799").location.end

            # MIBiG entry of BGC0001381 contains the whole contig;
            # authors don't really identify the minimal BGC but mark most
            # genes after `nbrU` as hypothetical or miscellaneous and most
            # genes before `nbrT3` as regulatory or miscellaneous
            # (see https://pubmed.ncbi.nlm.nih.gov/26754528/)
            elif record.id == "BGC0001381":
                start = get_cds(record, gene="nbrT3").location.start
                end = get_cds(record, gene="nbrU").location.end

            # MIBiG entry of BGC0001590 contains the whole antiSMASH prediction;
            # authors propose more stringent boundaries:
            # > Further bioinformatics analysis and consideration of the biosynthetic
            # > pathway leads us to propose that forQ and forCC represent the boundaries
            # > of BGC30
            # (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5414599/)
            elif record.id == "BGC0001590":
                start = get_cds(record, locus_tag="forQ").location.start
                end = get_cds(record, locus_tag="forCC").location.end

            # Heterologous expression of BGC0001917 shows that only `stmA` to `stmI`
            # are required for synthesis (see doi:10.1039/c8ob02846j, Fig.3)
            elif record.id == "BGC00001917":
                start = get_cds(record, gene="stmA").location.start
                end = get_cds(record, gene="stmI").location.end

            # MIBiG entry of BGC0001967 contains unrelated genes on both sides
            # of the `ade` operon
            elif record.id == "BGC0001967":
                start = get_cds(record, gene="adeA").location.start
                end = get_cds(record, gene="adeI").location.end

            # MIBiG entry of BGC0002043 contains many unrelated genes: the paper
            # performed heterologous expression and identified 3 core genes
            # + a distant peptidase
            # (see https://pubmed.ncbi.nlm.nih.gov/15600304/)
            elif record.id == "BGC0002043":
                start = get_cds(record, protein_id="QLH55578.1").location.start
                end = get_cds(record, protein_id="QLH55580.1").location.end

            # MIBiG entry of BGC0002087 contains the whole antiSMASH prediction
            # and not just the BGC homologous to the migrastatin BGC
            # (see https://onlinelibrary.wiley.com/doi/10.1002/anie.202009007)
            elif record.id == "BGC0002087":
                start = get_cds(record, protein_id="WP_025099964.1").location.start
                end = get_cds(record, protein_id="WP_036055450.1").location.end

            # MIBiG entry of BGC0002386 contains many unrelated genes; the paper
            # performed heterologous expression and identified 7 core genes,
            # although the whole synthesis seems to be done by the core NRPS
            # (see doi:10.12211/2096-8280.2021-024, Fig.2 and Fig.6)
            elif record.id == "BGC0002386":
                start = get_cds(record, locus_tag="SCE1572_24700").location.start
                end = get_cds(record, locus_tag="SCE1572_24730").location.end

            # MIBiG entry of BGC0002520 contains additional flanking genes
            # that have not been included in the BGC by the authors
            elif record.id == "BGC0002520":
                start = get_cds(record, protein_id="BCK51620.1").location.start
                end = get_cds(record, protein_id="BCK51663.1").location.end

            # MIBiG entry of BGC0002523 contains unrelated flanking genes;
            # supplementary material of the paper describes the core cluster
            # of going only from `dstR` to `dstG`
            # (see https://pubmed.ncbi.nlm.nih.gov/32457441/)
            elif record.id == "BGC0002523":
                start = get_cds(record, gene="dstR").location.start
                end = get_cds(record, gene="dstG").location.end

            # MIBiG entry of BGC0002676 contains unrelated genes downstream
            # of the BGC; two different papers about the skyllamycin BGC
            # limit the cluster at the DNA polIII gene
            # (see https://pubs.acs.org/doi/10.1021/acs.jnatprod.1c00547, Table S1;
            # and https://pubmed.ncbi.nlm.nih.gov/21456593/, Table 1)
            elif record.id == "BGC0002676":
                start = get_cds(record, locus_tag="KZO11_19415").location.start
                end = get_cds(record, locus_tag="KZO11_19670").location.end

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
