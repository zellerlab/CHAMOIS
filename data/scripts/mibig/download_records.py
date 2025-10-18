import argparse
import urllib.request
import os
import io
import tarfile

import rich.progress
import pandas
import Bio.SeqIO
import Bio.Entrez
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--blocklist")
parser.add_argument("--mibig-version", default="3.1", choices={"1.3", "2.0", "3.1", "4.0"})
parser.add_argument("--cache")
parser.add_argument("--email", default="martin.larralde@embl.de")
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

Bio.Entrez.email = args.email

# create persistent cache
if args.cache:
    rich.print(f"[bold green]{'Using':>12}[/] joblib cache folder {args.cache!r}")
    os.makedirs(args.cache, exist_ok=True)
memory = joblib.Memory(location=args.cache, verbose=False)

# download MIBiG requested version
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


@memory.cache
def get_genbank_record(genbank_id):
    with Bio.Entrez.efetch(db="nucleotide", id=genbank_id, rettype="gb", retmode="text") as f:
        return Bio.SeqIO.read(f, "genbank")


# a short sequence that ensures a STOP codon on all 6 frames
FULLSTOP = "TTAATTAATTAA"

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

            # BGC0000016 as found in MIBiG is only a fragment of the BGC
            # (https://github.com/mibig-secmet/mibig-json/issues/115) --
            # we recover the two parts and glue them to have a single record
            # with the whole cluster
            elif record.id == "BGC0000016":
                subcluster_1 = get_genbank_record("AY636001.1")
                subcluster_2 = get_genbank_record("AF357202.1")
                record = subcluster_1 + FULLSTOP + subcluster_2
                start = 0
                end = len(record)
                record.id = "BGC0000016"

            # BGC0000017 has unneeded downstream genes, only keep until `anaH`
            # (genes are characterized in doi:10.1016/j.toxicon.2014.07.016)
            elif record.id == "BGC0000017":
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

            # BGC0000081 has 33 unrelated genes upstream of the biosynthetic
            # core (see PMC3711097)
            elif record.id == "BGC0000081":
                start = get_cds(record, protein_id="AFV52131.1").location.start
                end = get_cds(record, protein_id="AFV52212.1").location.end

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

            # BGC0000136 has unrelated upstream and downstream genes, authors
            # show by heterologous expression and inactivation studies the AHBA
            # core is rifA-J (see PMID:11278540)
            elif record.id == "BGC0000136":
                start = get_cds(record, gene="rifA").location.start
                end = get_cds(record, gene="rifJ").location.end

            # BGC0000163 has unrelated genes, the deposited record contains
            # the core biosynthetic gene cluster as a feature
            elif record.id == "BGC0000163":
                start = get_cds(record, gene="tsn1").location.start
                end = get_cds(record, gene="tsn25").location.end

            # BGC0000187 has additional genes, the core BGC is only composed
            # of the 36 asu genes, shown by heterologous expression
            # (see PMID:20522559)
            elif record.id == "BGC0000187":
                start = get_cds(record, gene="asuR4").location.start
                end = get_cds(record, gene="asuR6").location.end

            # BGC0000213 ranges from colH1 to colR3, the genes downstream of
            # the cluster were disrupted without effect on colabomycin production
            # (see PMID:24838618)
            elif record.id == "BGC0000213":
                start = get_cds(record, gene="colH1").location.start
                end = get_cds(record, gene="colR3").location.end

            # BGC0000236 has extra upstream genes
            elif record.id == "BGC0000236":
                start = get_cds(record, gene="kinR").location.start
                end = get_cds(record, gene="kinX2").location.end

            # BGC0000268 has extra downstream genes
            # (https://www.sciencedirect.com/science/article/pii/S1016847823174048, Table 1)
            elif record.id == "BGC0000268":
                start = get_cds(record, gene="schA1").location.start
                end = get_cds(record, protein_id="CAH10129.1").location.end

            # BGC0000283 has extra flanking genes predicted to be unrelated
            # (see PMID:19101977)
            elif record.id == "BGC0000283":
                start = get_cds(record, gene="cetG").location.start
                end = get_cds(record, gene="cetF2").location.end

            # BGC0000291 has unclear boundaries, authors speculate up to
            # ORF17 until ORF46 (see doi:10.1007/s10295-005-0028-5)
            elif record.id == "BGC0000291":
                start = get_cds(record, protein_id="AAZ23059.1").location.start
                end = get_cds(record, protein_id="AAZ23088.1").location.end

            # BGC0000334 has extra flanking genes predicted to be unrelated
            # (see PMID:23818858)
            elif record.id == "BGC0000334":
                start = get_cds(record, locus_tag="simA").location.start
                end = get_cds(record, locus_tag="simL").location.end

            # BGC0000341 has unclear boundaries authors, speculate from
            # ORF22 to ORF46 (see PMID:17005978)
            elif record.id == "BGC0000341":
                start = get_cds(record, protein_id="ABD65942.1").location.start
                end = get_cds(record, protein_id="ABD65966.1").location.end

            # BGC0000379 has unclear boundaries, authors speculate from
            # ORF13 to ORF33 (see PMID:21640802)
            elif record.id == "BGC0000379":
                start = get_cds(record, protein_id="AEG64684.1").location.start
                end = get_cds(record, protein_id="AEF16032.1").location.end

            # BGC0000386 has incorrect locus prior to MIBiG 3.1
            # (https://github.com/mibig-secmet/mibig-json/issues/030)
            elif record.id == "BGC0000386":
                record = get_genbank_record("CP000086.1")
                record.id = record.name = "BGC0000386"
                start = get_cds(record, locus_tag="BTH_I2414").location.start
                end = get_cds(record, locus_tag="BTH_I2427").location.end

            # BGC0000409 has two unrelated genes on each end
            # (see PMID:25128200)
            elif record.id == "BGC0000409":
                start = get_cds(record, protein_id="AEA29624.1").location.start
                end = get_cds(record, protein_id="AEA29650.1").location.end

            # BGC0000413 contains many unrelated genes, so we can build the
            # minimum cluster using the reference paper
            # (https://pmc.ncbi.nlm.nih.gov/articles/PMC3562110/, Figure 2)
            elif record.id == "BGC0000413":
                record = get_genbank_record("CP000076.1")
                locus_tags = [
                    "PFL_4191", # pvdY
                    "PFL_4190", # pvdS
                    "PFL_4189", # pvdL
                    "PFL_4179", # pvdH
                    "PFL_4095", # pvdI
                    "PFL_4094", # pvdJ
                    "PFL_4093", # pvdD
                    "PFL_4091", # pvdE
                    "PFL_4090", # pvdF
                    "PFL_4089", # pvdO
                    "PFL_4088", # pvdN
                    "PFL_4087", # pvdM
                    "PFL_4086", # pvdP
                    "PFL_4082", # pvdT
                    "PFL_4081", # pvdR
                    "PFL_4079", # pvdA
                    "PFL_2902", # pvdQ
                ]
                features = [ get_cds(record, locus_tag=t).location.extract(record) for t in locus_tags ]
                record = features[0]
                for f in features[1:]:
                    record += FULLSTOP + f
                record.id = record.name = "BGC0000413"

            # BGC0000422 goes from sfmR1 to sfmO6, orf(-1) and orf(+1) outside
            # of cluster as shown by knock-out (see doi:10.1128/JB.00826-07)
            elif record.id == "BGC0000422":
                start = get_cds(record, gene="sfmR1").location.start
                end = get_cds(record, gene="sfmO6").location.end

            # BGC0000445 goes from tioA to tioZ, additional genes are transposases
            # or heat shock proteins (see PMID:16408310)
            elif record.id == "BGC0000445":
                start = get_cds(record, gene="tioA").location.start
                end = get_cds(record, gene="tioZ").location.end

            # BGC0000495 contains additional genes unrelated to actagardine
            # biosynthesis (see PMID:19400806)
            elif record.id == "BGC0000495":
                start = get_cds(record, gene="garA").location.start
                end = get_cds(record, gene="garR").location.end

            # BGC0000547 contains flanking genes from the insertion site
            # (see PMID:21310787, Fig. 1)
            elif record.id == "BGC0000547":
                start = get_cds(record, gene="gtfD").location.end + 1             # from after gtfD
                end = get_cds(record, protein_id="ACX68649.1").location.start - 1 # until before gtfX

            # BGC0000607 contains the full plasmid but only half the genes are
            # involved in micrococcin P1 biosynthesis (see PMID:25313391, Fig. 3)
            elif record.id == "BGC0000607":
                start = get_cds(record, gene="tclE").location.start
                end = get_cds(record, gene="tclU").location.end

            # BGC0000651 entry in MIBiG contains non-related flanking genes
            # on each end (see PMID:18071268, Fig. 2)
            elif record.id == "BGC0000651":
                start = get_cds(record, protein_id="BAF98618.1").location.start
                end = get_cds(record, protein_id="BAF98641.1").location.end

            # BGC0000653 entry in MIBiG contains non-related flanking genes
            # on each end (see PMID:21284395, Supplementary Scheme 1)
            elif record.id == "BGC0000653":
                start = get_cds(record, gene="pntR").location.start
                end = get_cds(record, gene="pntI").location.end

            # BGC0000684 only requires 3 central genes according to deletion
            # analyses (see PMID:18978088, Fig. 3)
            elif record.id == "BGC0000684":
                start = get_cds(record, locus_tag="ANIA_06002").location.start
                end = get_cds(record, locus_tag="ANIA_06000").location.end

            # BGC0000695 contains unrelated flanking genes on MIBiG
            # (unclear, source unpublished)
            elif record.id == "BGC0000695":
                start = get_cds(record, gene="forY").location.start
                end = get_cds(record, gene="fosA").location.end

            # BGC0000699 contains unrelated flanking genes on MIBiG
            # (see doi:10.1002/9780470149676.ch2, Table 2.22, and
            # https://pmc.ncbi.nlm.nih.gov/articles/PMC5856511/, Fig. 5)
            elif record.id == "BGC0000699":
                start = get_cds(record, gene="hygV").location.start
                end = get_cds(record, gene="hygG").location.end

            # BGC0000722 contains unrelated flanking genes on MIBiG, as
            # determined by deletion (see PMID:16632251, Fig. 2)
            elif record.id == "BGC0000722":
                start = get_cds(record, gene="valN").location.start
                end = get_cds(record, gene="valQ").location.end

            # BGC0000805 has a main cluster that produces the 
            # moenomycin phosphoglycolipid, and a subcluster that 
            # produces the C5N chromophore
            # (https://github.com/mibig-secmet/mibig-json/issues/092)
            # (NB: retired in MIBiG 4.0 anyway)
            elif record.id == "BGC0000805":
                cluster1 = get_genbank_record("DQ988994.1")
                cluster2 = get_genbank_record("DQ988993.1")
                start1 = get_cds(cluster1, gene="moeB5").location.start
                end1 = get_cds(cluster1, gene="moeS5").location.end
                start2 = get_cds(cluster2, gene="moeA4").location.start
                end2 = get_cds(cluster2, gene="moeC4").location.end
                assert start1 < end1
                assert start2 < end2
                record = cluster1[start1:end1] + FULLSTOP + cluster2[start2:end2]
                record.id = record.name = "BGC0000805"

            # BGC0000809 postulated by authors to go from atmA to atmI
            # based on GC% (see PMID:16873021)
            elif record.id == "BGC0000809":
                start = get_cds(record, gene="atA").location.start
                end = get_cds(record, gene="atI").location.end

            # BGC0000814 contains unrelated genes, authors show nokABCD is
            # sufficient for the carbazole core, nokP and additional for the
            # final compound, but many other genes are regulatory
            # (see PMID:19756308, Fig. 3)
            elif record.id == "BGC0000814":
                start = get_cds(record, protein_id="ACN29705.1").location.start  # nokX
                end = get_cds(record, protein_id="ACN29728.1").location.end # nokW

            # BGC0000874 contains unrelated genes at the beginning of the
            # record, which were shown to be unrelated to the biosynthesis
            # by restriction analysis (see PMID:12964155, Fig. 1)
            elif record.id == "BGC0000874":
                start = get_cds(record, gene="blsS").location.start
                end = get_cds(record, gene="blsN").location.end

            # BGC0000877 was shown to go from polB to polR by deletion
            # experiments (see PMID:19233844)
            elif record.id == "BGC0000877":
                start = get_cds(record, gene="polB").location.start
                end = get_cds(record, gene="polR").location.end

            # BGC0000938 has many uneeded genes, some of which have been shown
            # to be unneeded for biosynthesis through deletion analysis, and
            # the authors only consider fom3-fomC to be the core fosfomycin BGC
            # (see doi:10.1016/j.chembiol.2006.09.007)
            elif record.id == "BGC0000938":
                start = get_cds(record, gene="fom3").location.start
                end = get_cds(record, gene="fomC").location.end

            # BGC0000953 has some flanking genes
            # (see PMID:22267658, Fig. 3)
            elif record.id == "BGC0000953":
                start = get_cds(record, gene="amiA").location.start
                end = get_cds(record, gene="amiU").location.end

            # BGC0000966 has some flanking genes in the cluster that are not
            # counted as part of the BGC by the authors
            # (see https://pubmed.ncbi.nlm.nih.gov/22591508/, Figure 1)
            elif record.id == "BGC0000966":
                start = get_cds(record, protein_id="AFD30965.1").location.start
                end = get_cds(record, protein_id="AFD30946.1").location.end

            # BGC0001010 has some unrelated upstream genes
            # (see https://pubmed.ncbi.nlm.nih.gov/14583260/, Table 1)
            elif record.id == "BGC0001010":
                start = get_cds(record, gene="melB").location.start
                end = get_cds(record, gene="melK").location.end

            # BGC0001073 contains some unrelated genes
            # (see https://pubmed.ncbi.nlm.nih.gov/23790490/, Table 1)
            elif record.id == "BGC0001073":
                start = get_cds(record, protein_id="AFJ52659.1").location.start
                end = get_cds(record, protein_id="AFJ52701.1").location.end

            # MIBiG entry contains flanking genes that were shown to be
            # unrelated by inactivation experiments (see PMID:23913777, Fig. 1)
            elif record.id == "BGC0001137":
                start = get_cds(record, protein_id="AGL76720.1").location.start
                end = get_cds(record, protein_id="AGL76722.1").location.end

            # MIBiG entry contains genes reported by the authors to be outside
            # the cluster (however with no experimental confirmation, see PMID:22064543)
            elif record.id == "BGC0001138":
                start = get_cds(record, protein_id="AET51839.1").location.start
                end = get_cds(record, protein_id="AET51866.1").location.end

            # MIBiG entry contains unrelated genes, authors list lobR1 to lobR4
            # (see PMID:27005505, Supplementary Table 1)
            elif record.id == "BGC0001183":
                start = get_cds(record, protein_id="AGC09475.1").location.start
                end = get_cds(record, protein_id="AGC09510.1").location.end

            # MIBiG entry misses genes `ywfH` and `ywfI`
            # (https://github.com/mibig-secmet/mibig-json/issues/400)
            elif record.id == "BGC0001184":
                record = get_genbank_record("CP000560.1")
                record.id = record.name = "BGC0001184"
                start = get_cds(record, gene="ywfI").location.start
                end = get_cds(record, gene="ywfA").location.end

            # MIBIG entry of BGC0001196 contains ORF-2 to ORF+3
            elif record.id == "BGC0001196":
                start = get_cds(record, protein_id="AJW76703.1").location.start
                end = get_cds(record, protein_id="AJW76719.1").location.end

            # MIBiG entry of BGC0001202 contains unrelated upstream and
            # downstream genes
            elif record.id == "BGC0001202":
                start = get_cds(record, protein_id="AKA59430.1").location.start
                end = get_cds(record, protein_id="AKA59442.1").location.end

            # BGC0001300 spans from atcC to atcI, whole biosynthesis
            # described without additional genes (see PMID:26348978, Fig. 3)
            elif record.id == "BGC0001300":
                start = get_cds(record, gene="atcC").location.start
                end = get_cds(record, gene="atcI").location.end

            # BGC0001344 was characterized with heterologous expression of
            # genes tubA to orf18 (see PMID:22444591, Fig. 1)
            elif record.id == "BGC0001344":
                start = get_cds(record, gene="tubA").location.start
                end = get_cds(record, protein_id="ADH04684.1").location.end  # orf18

            # BGC0001367 has unneeded downstream genes, the authors only
            # consider the two core genes `hla1` and `hla2` to be part of
            # the BGC and sufficient for synthesis of the final compound
            # (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6274090/)
            elif record.id == "BGC0001367":
                start = get_cds(record, locus_tag="Hoch_0798").location.start
                end = get_cds(record, locus_tag="Hoch_0799").location.end

            # BGC0001379 has two flanking genes that are not part of the
            # cluster, as shown by heterologous expression
            # (see PMID:27084005, Fig. 2)
            elif record.id == "BGC0001379":
                start = get_cds(record, protein_id="BAU50925.1").location.start
                end = get_cds(record, protein_id="BAU50950.1").location.end

            # MIBiG entry of BGC0001381 contains the whole contig;
            # authors don't really identify the minimal BGC but mark most
            # genes after `nbrU` as hypothetical or miscellaneous and most
            # genes before `nbrT3` as regulatory or miscellaneous
            # (see https://pubmed.ncbi.nlm.nih.gov/26754528/)
            elif record.id == "BGC0001381":
                start = get_cds(record, gene="nbrT3").location.start
                end = get_cds(record, gene="nbrU").location.end

            # BGC0001440 and BGC0001441 are homologous, BGC0001440 spans
            # cysA to cysH as confirmed by inactivation studies, BGC0001441
            # likely spans belA to belV based on homology
            # (see https://pubmed.ncbi.nlm.nih.gov/28452105/, Fig. 3)
            elif record.id == "BGC0001440":
                start = get_cds(record, gene="cysA").location.start
                end = get_cds(record, gene="cysH").location.end
            elif record.id == "BGC0001441":
                start = get_cds(record, gene="belA").location.start
                end = get_cds(record, gene="belV").location.end

            # BGC0001457 coordinates spanning at most orf8 to orf24, confirmed
            # by heterologous expression
            # (see https://pubmed.ncbi.nlm.nih.gov/30065171/, Fig. 1)
            elif record.id == "BGC0001457":
                start = get_cds(record, locus_tag="ctg1_orf8").location.start
                end = get_cds(record, locus_tag="ctg1_orf24").location.end

            # BGC0001522 reported by the authors to span from aurR1 to aurT4
            # (see doi:10.1002/cbic.201800266, Table S4)
            elif record.id == "BGC0001522":
                start = get_cds(record, protein_id="AWR88389.1").location.start
                end = get_cds(record, protein_id="AWR88426.1").location.end

            # BGC0001540 spans from c10R1 to c10R6 according to the authors
            # (see https://pubmed.ncbi.nlm.nih.gov/28426198/, Table 1, Fig. 2)
            elif record.id == "BGC0001540":
                start = get_cds(record, gene="c10R1").location.start
                end = get_cds(record, gene="c10R6").location.end

            # BGC0001575 was characterized by heterologous expression of
            # the 4 terminal genes only
            # (see doi:10.1016/j.cell.2016.12.021, Fig. S1, Fig. 2)
            elif record.id == "BGC0001575":
                start = get_cds(record, locus_tag="RSAG_RS12300").location.start
                end = get_cds(record, locus_tag="RSAG_RS12315").location.end

            # MIBiG entry of BGC0001590 contains the whole antiSMASH prediction;
            # authors propose more stringent boundaries:
            # > Further bioinformatics analysis and consideration of the biosynthetic
            # > pathway leads us to propose that forQ and forCC represent the boundaries
            # > of BGC30
            # (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5414599/)
            elif record.id == "BGC0001590":
                start = get_cds(record, locus_tag="forQ").location.start
                end = get_cds(record, locus_tag="forCC").location.end

            # MIBiG entry of BGC0001619 has incorrect locus prior to MIBiG 4.0
            # (https://github.com/mibig-secmet/mibig-json/issues/161)
            elif record.id == "BGC0001619":
                record = get_genbank_record("KJ159185.1")
                record.id = record.name = "BGC0001619"
                start = get_cds(record, protein_id="AKA59073.1").location.start
                end = get_cds(record, protein_id="AKA59111.1").location.end

            # MIBiG entry contains additional flanking genes not part of the
            # cluster according to sequence homology
            # (see https://pmc.ncbi.nlm.nih.gov/articles/PMC11959601/, Fig. 1)
            elif record.id == "BGC0001728":
                start = get_cds(record, locus_tag="PPE_RS11465").location.start
                end = get_cds(record, locus_tag="PPE_RS11500").location.end

            # BGC0001731 contains flanking genes, RT-PCR experiments show the
            # cluster boundaries to be plm2 (pau2) to plm42 (pau42)
            # (see https://pubmed.ncbi.nlm.nih.gov/27001601, Fig. 2)
            elif record.id == "BGC0001731":
                start = get_cds(record, product="Pau2").location.start # pau2
                end = get_cds(record, product="Pau42").location.end  # pau42

            # BGC0001735 core cluster is at most orf(-1) to orf(+1), as shown
            # by heterologous expression
            # (see https://pubmed.ncbi.nlm.nih.gov/28111097/, Fig. 1)
            elif record.id == "BGC0001735":
                start = get_cds(record, protein_id="AKA87333.1").location.start
                end = get_cds(record, protein_id="AOE46836.1").location.end

            # BGC0001751 contains extra upstream transposases and downstream
            # sugar-processing enzymes not part of the cluster as per the
            # authors (see https://pubmed.ncbi.nlm.nih.gov/28590072/, Fig. 3)
            elif record.id == "BGC0001751":
                start = get_cds(record, gene="pyxD").location.start
                end = get_cds(record, gene="pyxT").location.end

            # BGC0001759 contains flanking DNA polymerase and ribosomal proteins
            # that are not discussed by the authors
            # (see https://pubmed.ncbi.nlm.nih.gov/28422262/, Scheme 1)
            elif record.id == "BGC0001759":
                start = get_cds(record, product="Rmp36").location.start
                end = get_cds(record, product="Rmp21").location.end

            # BGC0001762 contains additional gene that are outside the cluster
            # as shown by different cosmids
            # (see https://www.nature.com/articles/ncomms13083, Fig. 3)
            elif record.id == "BGC0001762":
                start = get_cds(record, gene="rubS1").location.start
                end = get_cds(record, gene="rubE9").location.end

            # BGC0001774 contains unrelated downstream genes, as shown by
            # inactivation experiments on sepM-Q
            # (see https://pmc.ncbi.nlm.nih.gov/articles/PMC5856511/, Fig. 2)
            elif record.id == "BGC0001774":
                start = get_cds(record, gene="sepR").location.start
                end = get_cds(record, gene="sepL").location.end

            # BGC0001885 contains an initial repressor gene we can safely
            # exclude from the BGC
            # (see https://pubmed.ncbi.nlm.nih.gov/30598544/)
            elif record.id == "BGC0001885":
                start = get_cds(record, gene="ozeA").location.start
                end = get_cds(record, gene="ozeN").location.end

            # BGC0001817 describes 5 genes that are apart in a contig, but
            # all relate to the same pathway, we create a composite cluster
            # merging all genes in a single record
            elif record.id == "BGC0001817":
                proteins = ["AKA59112.1", "AKA59113.1", "AKA59114.1", "AKA59115.1", "AKA59116.1"]
                features = [ get_cds(record, protein_id=p).location.extract(record) for p in proteins ]
                record = features[0]
                for f in features[1:]:
                    record += FULLSTOP + f
                record.id = record.name = "BGC0001817"

            # BGC0001891 has wrong locus coordinates prior to MIBiG 4.0
            # (https://github.com/mibig-secmet/mibig-json/issues/037)
            elif record.id == "BGC0001891":
                record = get_genbank_record("CP001348.1")
                record.id = record.name = "BGC0001891"
                start = get_cds(record, locus_tag="Ccel_3250").location.start
                end = get_cds(record, locus_tag="Ccel_3260").location.end

            # Heterologous expression of BGC0001917 shows that only `stmA` to `stmI`
            # are required for synthesis (see doi:10.1039/c8ob02846j, Fig.3)
            elif record.id == "BGC00001917":
                start = get_cds(record, gene="stmA").location.start
                end = get_cds(record, gene="stmI").location.end

            # LC-MS shows that BGC0001945 produces chloromyxamide A-E
            # (see https://pubmed.ncbi.nlm.nih.gov/30088846/, Fig.1)
            elif record.id == "BGC0001945":
                start = get_cds(record, gene="cmxA").location.start
                end = get_cds(record, gene="cmxS").location.end

            # MIBiG entry of BGC0001967 contains unrelated genes on both sides
            # of the `ade` operon
            elif record.id == "BGC0001967":
                start = get_cds(record, gene="adeA").location.start
                end = get_cds(record, gene="adeI").location.end

            # MIBiG entry of BGC0001992 is using the wrong GenBank locus
            # (https://github.com/mibig-secmet/mibig-json/issues/065)
            elif record.id == "BGC0001992":
                record = get_genbank_record("FN666575.1")
                record.id = record.name = "BGC0001992"
                start = get_cds(record, locus_tag="EAM_1028").location.start
                end = get_cds(record, locus_tag="EAM_1033").location.end

            # MIBiG entry has the whole cluster, but authors through inactivation
            # experiments show that the biosynthetic core is atoA-F
            # (see https://pmc.ncbi.nlm.nih.gov/articles/PMC6547381/, Fig. 3)
            elif record.id == "BGC0002007":
                start = get_cds(record, locus_tag="BLW76_RS36800").location.start # atoA
                end = get_cds(record, locus_tag="BLW76_RS36825").location.end # atoF

            # MIBiG entry contains unrelated flanking genes
            # (see https://dx.doi.org/10.1021/acs.orglett.9b01891, Fig. 1)
            elif record.id == "BGC0002011":
                start = get_cds(record, locus_tag="AMYAL_RS47350").location.start
                end = get_cds(record, locus_tag="AMYAL_RS0139605").location.end

            # MIBiG entry contains two additional flanking genes
            # (see https://journals.asm.org/doi/10.1128/aem.01971-19, Fig. 2)
            elif record.id == "BGC0002039":
                start = get_cds(record, gene="focS").location.start
                end = get_cds(record, gene="focP").location.end

            # MIBiG entry contains larger region than determined by the
            # authors (although experimental validation does not show
            # cluster completeness, see https://www.mdpi.com/1660-3397/19/4/209)
            elif record.id == "BGC0002044":
                start = get_cds(record, locus_tag="DPH57_09130").location.start
                end = get_cds(record, locus_tag="DPH57_09150").location.end

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

            # MIBiG entry has the whole contig expressed heterologously, but
            # sequence homology with BGC0002199 suggests that only two
            # genes homologous to `nanA` and `nanC` are involved
            # (see https://pubmed.ncbi.nlm.nih.gov/10.1021/jacs.0c01605/, Fig. 7)
            elif record.id == "BGC0002198":
                start = get_cds(record, gene="FAC38_35").location.start
                end = get_cds(record, gene="FAC38_37").location.start-1 # to account for introns

            # BGC0002359 entry has an entire fragment but authors confirmed
            # by deletion experiments and heterologous expression that genes
            # orf1-orf21 and orf25-26 are not involved in biosynthesis
            # (see https://www.mdpi.com/2076-2607/8/11/1800, Fig. 4)
            elif record.id == "BGC0002359":
                start = get_cds(record, locus_tag="FM076_21195").location.start
                end = get_cds(record, locus_tag="FM076_21205").location.end

            # BGC0002373 entry limits from bnvA to bnvO according to the authors
            # (see doi:10.1021/acs.joc.1c00360, Fig. 8)
            elif record.id == "BGC0002373":
                start = get_cds(record, gene="bnvO").location.start
                end = get_cds(record, gene="bnvA").location.end

            # BGC0002379 contains two BGC halves separated by 70kb; we merge
            # the two halves to obtain the minimum BGC reported by the authors
            # (https://pubmed.ncbi.nlm.nih.gov/10.1002/anie.202015193/, Fig. 4)
            elif record.id == "BGC0002379":
                record = get_genbank_record("AP023348.2")
                start1 = get_cds(record, protein_id="BDU09799.1").location.start
                end1 = get_cds(record, protein_id="BCJ07527.1").location.end
                start2 = get_cds(record, protein_id="BCJ07604.1").location.start
                end2 = get_cds(record, protein_id="BDU09794.1").location.end
                assert start1 < end1
                assert start2 < end2
                record = record[start1:end1] + FULLSTOP + record[start2:end2]
                record.id = record.name = "BGC0002379"

            # BGC0002384 is actually only formed of 4 genes
            # (see doi:10.1021/acssynbio.0c00067, Supplementary Table 2)
            elif record.id == "BGC0002384":
                start = get_cds(record, protein_id="QWM97318.1").location.start
                end = get_cds(record, protein_id="QWM97321.1").location.end

            # MIBiG entry of BGC0002386 contains many unrelated genes; the paper
            # performed heterologous expression and identified 7 core genes,
            # although the whole synthesis seems to be done by the core NRPS
            # (see doi:10.12211/2096-8280.2021-024, Fig.2 and Fig.6)
            elif record.id == "BGC0002386":
                start = get_cds(record, locus_tag="SCE1572_24700").location.start
                end = get_cds(record, locus_tag="SCE1572_24730").location.end

            # MIBiG entry of BGC0002387 contains unrelated upstream genes,
            # and is likely missing downstream genes that are likely not
            # biosynthetic (see https://pmc.ncbi.nlm.nih.gov/articles/PMC6204215/,
            # Supplementary Table S1)
            elif record.id == "BGC0002387":
                start = get_cds(record, locus_tag="GA0070617_3550").location.start
                end = get_cds(record, locus_tag="GA0070617_3572").location.end

            # BGC0002409 and BGC0002410 are two halves of the frankobactin
            # cluster (while a complete cluster can be found in Frankia sp. DC12
            # but has not been validated) -- we merge the two halves inside
            # BGC0002409 and add BGC0002410 to the blocklist to create a single
            # composite BGC
            # (see https://pubs.acs.org/doi/10.1021/acs.jnatprod.0c01291, Fig. S18-S19)
            elif record.id == "BGC0002409":
                record = get_genbank_record("JADBID010000001.1")
                start1 = get_cds(record, locus_tag="IHE48_05525").location.start
                end1 = get_cds(record, locus_tag="IHE48_05595").location.end
                start2 = get_cds(record, locus_tag="IHE48_18560").location.start
                end2 = get_cds(record, locus_tag="IHE48_18615").location.end
                assert start1 < end1
                assert start2 < end2
                record = record[start1:end1] + FULLSTOP + record[start2:end2]
                record.id = record.name = "BGC0002409"

            # BGC0002426 spans from babR1 to babR8
            # (see doi:10.1039/D1OB00600B, Table S22)
            elif record.id == "BGC0002426":
                start = get_cds(record, gene="babR1").location.start
                end = get_cds(record, gene="babR8").location.end

            # BGC0002432 is almost identical to BGC0001536 (brevicidine),
            # we can resize the two clusters to their homologous proteins
            # (see https://pubmed.ncbi.nlm.nih.gov/30115920/, Fig. 3)
            elif record.id == "BGC0002432":
                start = get_cds(record, locus_tag="BRLA_c025570").location.start
                end = get_cds(record, locus_tag="BRLA_c025670").location.end

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

            # MIBiG entry contains extra flanking genes
            # (see https://pubmed.ncbi.nlm.nih.gov/31756295/, Figure 4)
            elif record.id == "BGC0002558":
                start = get_cds(record, gene="sprD").location.start
                end = get_cds(record, protein_id="BBM95968.1").location.end

            # MIBiG entry contains extra flanking genes
            # (see https://pmc.ncbi.nlm.nih.gov/articles/PMC9136965/, Figure 2)
            elif record.id == "BGC0002637":
                start = get_cds(record, locus_tag="DMH18_08025").location.start
                end = get_cds(record, locus_tag="DMH18_08120").location.end

            # MIBiG entry contains extra flanking genes
            # (see https://www.mdpi.com/1660-3397/19/12/673, Fig. 1)
            elif record.id == "BGC0002671":
                start = get_cds(record, gene="kmy1").location.start
                end = get_cds(record, gene="kmy29").location.end

            # MIBiG entry of BGC0002676 contains unrelated genes downstream
            # of the BGC; two different papers about the skyllamycin BGC
            # limit the cluster at the DNA polIII gene
            # (see https://pubs.acs.org/doi/10.1021/acs.jnatprod.1c00547, Table S1;
            # and https://pubmed.ncbi.nlm.nih.gov/21456593/, Table 1)
            elif record.id == "BGC0002676":
                start = get_cds(record, locus_tag="KZO11_19415").location.start
                end = get_cds(record, locus_tag="KZO11_19670").location.end

            # BGC0002713 consists only in the core NRPs as shown by the
            # gene activation experiments
            # (see https://pubmed.ncbi.nlm.nih.gov/31611640/, Fig. 2)
            elif record.id == "BGC0002713":
                start = get_cds(record, locus_tag="PluTT01m_04610").location.start
                end = get_cds(record, locus_tag="PluTT01m_04620").location.end

            # MIBiG entry of BGC0002715 contains unrelated flanking genes, which
            # were not part of the cluster that was validated through heterologous
            # expression
            # (see https://pubmed.ncbi.nlm.nih.gov/18722414/, Table 1 and Figure 2)
            elif record.id == "BGC0002715":
                start = get_cds(record, locus_tag="PluTT01m_11950").location.start
                end = get_cds(record, locus_tag="PluTT01m_11995").location.end

            # MIBiG entry of BGC0002859 has the wrong locus
            # (see https://github.com/mibig-secmet/mibig-json/issues/403)
            elif record.id == "BGC0002859":
                record = get_genbank_record("UZVX01000002.1")
                record.id = record.name = "BGC0002859" 
                start = 413250  # determined by BLASTn
                end = 484630  # determined by BLASTn

            # BGC0002873 contains unrelated flanking genes not recognized by
            # the authors as part of the BGC
            elif record.id == "BGC0002873":
                start = get_cds(record, locus_tag="H9D14_RS04840").location.start # gbnR
                end = get_cds(record, locus_tag="H9D14_RS04960").location.end # gbnD6

            # MIBiG entry of BGC0002977 contains the entire contig, but the
            # heterologously cloned cluster coordinates can be recovered from
            # the paper
            # (https://www.mdpi.com/1420-3049/26/7/1892, Table S2-S4)
            elif record.id == "BGC0002977":
                record = get_genbank_record("LYMC01000002.1")
                record.id = record.name = "BGC0002977"
                start = 610900
                end = 655972

            # MIBiG entry of BGC0002919 contains unrelated flanking genes
            # (see https://doi.org/10.1002/chem.202400271 Figure 1)
            elif record.id == "BGC0002919":
                start = get_cds(record, gene="cirB").location.start
                end = get_cds(record, gene="cirA12").location.end

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
            if not "molecule_type" in bgc_record.annotations:
                bgc_record.annotations["molecule_type"] = "DNA"
            records.append(bgc_record)

    # sort records by MIBiG accession
    records.sort(key=lambda record: record.id)

    # save records
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as dst:
        Bio.SeqIO.write(records, dst, "genbank")
