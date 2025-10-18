import pathlib
import io
import gzip
import zipfile
import tempfile
import sys

import Bio.SeqIO
import pandas
import rich.progress

folder = pathlib.Path(__file__).absolute().parent
sys.path.insert(0, str(folder.parents[2]))

import chamois.cli

# load antiSMASH type translation map
type_map = pandas.read_table(folder.joinpath("chem_class_map.tsv"))
type_index = dict(zip(type_map["chem_code"].str.lower(), type_map["bigslice_class"]))

# load coordinates of native BGCs
coordinates = pandas.read_table(
    folder.parents[2].joinpath("data", "datasets", "native", "coordinates.tsv")
)

# Load GECCO predictions
rich.print(f"[bold green]{'Loading':>12}[/] GECCO predictions")
gecco_predictions = pandas.concat([
    pandas.read_table(filename).assign(source="gecco", genome_id=filename.name.rsplit(".", 2)[0])
    for filename in folder.parents[2].joinpath("data", "datasets", "native", "gecco").glob("*.clusters.tsv")
])
rich.print(f"[bold green]{'Loaded':>12}[/] {len(gecco_predictions)} BGC predictions in {gecco_predictions['genome_id'].nunique()} genomes")

# Load antiSMASH predictions
antismash_predictions_raw = []
rich.print(f"[bold green]{'Loading':>12}[/] antiSMASH predictions")
for archive in rich.progress.track(list(folder.parents[2].joinpath("data", "datasets", "native", "antismash7").glob("*.zip")), description=f"[bold blue]{'Working':>12}[/]"):
    with zipfile.ZipFile(archive) as zip_file:
        with zip_file.open(f"{archive.stem}.gbk") as src:
            # get all BGC regions
            records = Bio.SeqIO.parse(io.TextIOWrapper(src), "genbank")
            for record in records:
                for region in filter(lambda f: f.type == "region", record.features):
                    region_number = region.qualifiers['region_number'][0]
                    region_id = f"{record.id}_region{region_number}"
                    antismash_predictions_raw.append([
                        archive.stem,
                        record.id,
                        region_id,
                        region.location.start,
                        region.location.end,
                        ";".join(region.qualifiers['product']),
                        "antismash",
                    ])
antismash_predictions = pandas.DataFrame(
    antismash_predictions_raw,
    columns=["genome_id", "sequence_id", "cluster_id", "start", "end", "product", "source"],
)
rich.print(f"[bold green]{'Loaded':>12}[/] {len(antismash_predictions)} BGC predictions in {antismash_predictions['genome_id'].nunique()} genomes")

# merge the cluster sets with union / intersection
sequence_ids = set(antismash_predictions.sequence_id) | set(gecco_predictions.sequence_id)
unique_clusters_raw = []
for sequence_id in sorted(sequence_ids):
    sequence_antismash = antismash_predictions[antismash_predictions.sequence_id == sequence_id]
    sequence_gecco = gecco_predictions[gecco_predictions.sequence_id == sequence_id]
    for row in sequence_antismash.itertuples():
        overlaps = sequence_gecco[(sequence_gecco.start <= row.end) & (row.start <= sequence_gecco.end)]
        row_type = ";".join(sorted({ type_index[p.lower()] for p in row.product.split(";") }))
        if not overlaps.empty:
            start = overlaps.start.min()
            end = overlaps.end.max()
            unique_clusters_raw.append([
                row.genome_id,
                row.sequence_id,
                row.cluster_id + "_intersect",
                max(row.start, start),
                min(row.end, end),
                row.product,
                row_type,
                "intersect",
            ])
        else:
            unique_clusters_raw.append([
                row.genome_id,
                row.sequence_id,
                row.cluster_id,
                row.start,
                row.end,
                row.product,
                row_type,
                row.source,
            ])
    for row in sequence_gecco.itertuples():
        overlaps = sequence_antismash[(sequence_antismash.start <= row.end) & (row.start <= sequence_antismash.end)]
        if overlaps.empty: # overlaps handled above
            unique_clusters_raw.append([
                row.genome_id,
                row.sequence_id,
                row.cluster_id,
                row.start,
                row.end,
                None,
                row.type,
                row.source
            ])
    
for row in coordinates.itertuples():
    if not any(
        cluster[1] == row.sequence_id
        and cluster[3] <= row.end
        and row.start <= cluster[4]
        for cluster in unique_clusters_raw
    ):
        rich.print(f"[bold yellow]{'Adding':>12}[/] manual cluster prediction for [bold cyan]{row.bgc_id}[/]")
        unique_clusters_raw.append([
            row.genome,
            row.sequence_id,
            row.bgc_id,
            row.start,
            row.end,
            None,
            "Other",
            "Manual",
        ])

unique_clusters = pandas.DataFrame(
    unique_clusters_raw,
    columns=["genome_id", "sequence_id", "cluster_id", "start", "end", "product", "type", "source"]
)
unique_clusters.to_csv(
    folder.joinpath("merged.tsv"),
    index=False,
    sep="\t",
)

# extract clusters and run CHAMOIS predictions
clusters = []
for genome in rich.progress.track(list(folder.parents[2].joinpath("data", "datasets", "native", "genomes").glob("*.fna")), description=f"[bold blue]{'Working':>12}[/]"):
#for archive in rich.progress.track(list(folder.parents[1].joinpath("data", "datasets", "native", "antismash7").glob(f"*.zip")), description=f"[bold blue]{'Working':>12}[/]"):
    #with zipfile.ZipFile(archive) as zip_file:
        #with zip_file.open(f"{archive.stem}.gbk") as src:

            # get all BGC regions
    records = Bio.SeqIO.parse(genome, "fasta")
    for record in records:
        record_clusters = unique_clusters[unique_clusters.sequence_id == record.id]
        for row in record_clusters.itertuples():
            cluster = record[row.start:row.end]
            cluster.id = cluster.name = row.cluster_id
            cluster.annotations['molecule_type'] = "DNA"
            clusters.append(cluster)

Bio.SeqIO.write(clusters, folder.joinpath("merged.gbk"), "genbank")
chamois.cli.run([
    "predict", 
    "-i", str(folder.joinpath("merged.gbk")), 
    "-o", str(folder.joinpath("merged.hdf5")),
    "--model", str(folder.joinpath("predictor.mibig3.1.json")),
    "--hmm", str(folder.parents[2].joinpath("data", "pfam", "Pfam38.0.hmm")),
])
