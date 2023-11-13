EMAIL=$(shell git config user.email)

DATA=data
BUILD=build
SCRIPTS=$(DATA)/scripts

GO_VERSION=2023-06-11
GO_OBO=$(DATA)/ontologies/go$(GO_VERSION).obo

MIBIG=$(DATA)/mibig
MIBIG_VERSION=3.1

INTERPRO_VERSION=94.0
INTERPRO_XML=$(DATA)/pfam/interpro$(INTERPRO_VERSION).xml.gz
INTERPRO_JSON=$(DATA)/pfam/interpro$(INTERPRO_VERSION).json

PFAM_VERSION=36.0
PFAM_HMM=$(DATA)/pfam/Pfam$(PFAM_VERSION).hmm

ATLAS=$(DATA)/npatlas/NPAtlas_download.json.gz
CHEMONT=$(DATA)/ontologies/ChemOnt_2_1.obo

DATASET_NAMES=mibig3.1 mibig2.0
DATASET_TABLES=features classes ani

TAXONOMY=$(DATA)/taxonomy
TAXONOMY_VERSION=2023-06-01

PYTHON=python -Wignore
WGET=wget --no-check-certificate

.PHONY: datasets
datasets: features classes compounds clusters maccs

.PHONY: features
features: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/features.hdf5)

.PHONY: classes
classes: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/classes.hdf5)

.PHONY: maccs
maccs: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/maccs.hdf5)

.PHONY: compounds
compounds: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/compounds.json)

.PHONY: clusters
clusters: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/clusters.gbk)

.PHONY: ani
ani: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/ani.hdf5)


# --- External data ----------------------------------------------------------

$(PFAM_HMM):
	$(WGET) http://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam$(PFAM_VERSION)/Pfam-A.hmm.gz -O- | gunzip > $@

$(PGAP_HMM):
	$(WGET) ftp://ftp.ncbi.nlm.nih.gov/hmm/11.0/hmm_PGAP.LIB -O $@

$(SMCOGS_HMM):
	$(PYTHON) $(SCRIPTS)/smcogs/download.py --version $(SMCOGS_VERSION) --output $@

$(TAXONOMY)/taxdmp_$(TAXONOMY_VERSION).zip:
	mkdir -p $(TAXONOMY)
	$(WGET) https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump_archive/taxdmp_$(TAXONOMY_VERSION).zip -O $@

$(TAXONOMY)/%.dmp: $(TAXONOMY)/taxdmp_$(TAXONOMY_VERSION).zip
	unzip $< -d $(TAXONOMY) $(notdir $@)
	touch $@

# --- InterPro ---------------------------------------------------------------

$(GO_OBO):
	$(WGET) http://purl.obolibrary.org/obo/go/releases/$(GO_VERSION)/go.obo -O $@

$(INTERPRO_XML):
	$(WGET) https://ftp.ebi.ac.uk/pub/databases/interpro/releases/$(INTERPRO_VERSION)/interpro.xml.gz -O $@

$(INTERPRO_JSON): $(GO_OBO) $(INTERPRO_XML)
	python $(SCRIPTS)/pfam/interpro_json.py --go $(GO_OBO) --xml $(INTERPRO_XML) --out $@


# --- NP Atlas ---------------------------------------------------------------

$(ATLAS):
	$(WGET) https://www.npatlas.org/static/downloads/NPAtlas_download.json -O- | gzip -c > $@

$(DATA)/npatlas/classes.hdf5: $(CHEMONT) $(ATLAS) 
	$(PYTHON) $(SCRIPTS)/npatlas/make_classes.py --atlas $(ATLAS) --chemont $(CHEMONT) -o $@

$(DATA)/npatlas/maccs.hdf5: $(ATLAS) 
	$(PYTHON) $(SCRIPTS)/npatlas/make_maccs.py --atlas $(ATLAS) -o $@


# --- Generic Rules ----------------------------------------------------------

#$(DATA)/datasets/%/mibig3.1_ani.hdf5: $(DATA)/datasets/%/clusters.gbk $(DATA)/datasets/mibig3.1/clusters.gbk
#	$(PYTHON) $(SCRIPTS)/common/make_ani.py --query $< --target $(DATA)/datasets/mibig3.1/clusters.gbk -o $@

$(DATA)/datasets/%/features.hdf5: $(DATA)/datasets/%/clusters.gbk $(PFAM_HMM)
	$(PYTHON) -m conch.cli annotate --i $< --hmm $(PFAM_HMM) -o $@

$(DATA)/datasets/%/classes.hdf5: $(DATA)/datasets/%/compounds.json $(ATLAS) $(CHEMONT)
	$(PYTHON) $(SCRIPTS)/common/make_classes.py -i $< -o $@ --atlas $(ATLAS) --chemont $(CHEMONT) --cache $(BUILD)

$(DATA)/datasets/%/maccs.hdf5: $(DATA)/datasets/%/compounds.json $(ATLAS) $(CHEMONT)
	$(PYTHON) $(SCRIPTS)/common/make_maccs.py -i $< -o $@

$(DATA)/datasets/%/ani.hdf5: $(DATA)/datasets/%/clusters.gbk $(CHEMONT)
	$(PYTHON) $(SCRIPTS)/common/make_ani.py -i $< -o $@ -s 0.3


# --- Download MIBiG 2.0 data ------------------------------------------------

$(DATA)/datasets/mibig2.0/clusters.gbk: $(DATA)/mibig/blocklist.tsv $(SCRIPTS)/mibig/download_records.py
	$(PYTHON) $(SCRIPTS)/mibig/download_records.py --blocklist $< --mibig-version 2.0 -o $@

$(DATA)/datasets/mibig2.0/compounds.json: $(DATA)/mibig/blocklist.tsv $(ATLAS) $(SCRIPTS)/mibig/download_compounds.py
	$(PYTHON) $(SCRIPTS)/mibig/download_compounds.py --blocklist $< --mibig-version 2.0 -o $@ --atlas $(ATLAS) --cache $(BUILD)

$(DATA)/datasets/mibig2.0/taxonomy.tsv: $(DATA)/mibig/blocklist.tsv $(TAXONOMY)/names.dmp $(TAXONOMY)/nodes.dmp $(TAXONOMY)/merged.dmp
	$(PYTHON) $(SCRIPTS)/mibig/download_taxonomy.py --blocklist $< --mibig-version 2.0 -o $@ --taxonomy $(TAXONOMY)


# --- Download MIBiG 3.1 data ------------------------------------------------

$(DATA)/datasets/mibig3.1/clusters.gbk: $(DATA)/mibig/blocklist.tsv $(SCRIPTS)/mibig/download_records.py
	$(PYTHON) $(SCRIPTS)/mibig/download_records.py --blocklist $< --mibig-version 3.1 -o $@

$(DATA)/datasets/mibig3.1/compounds.json: $(DATA)/mibig/blocklist.tsv $(ATLAS) $(SCRIPTS)/mibig/download_compounds.py
	$(PYTHON) $(SCRIPTS)/mibig/download_compounds.py --blocklist $< --mibig-version 3.1 -o $@ --atlas $(ATLAS) --cache $(BUILD)

$(DATA)/datasets/mibig3.1/taxonomy.tsv: $(DATA)/mibig/blocklist.tsv $(TAXONOMY)/names.dmp $(TAXONOMY)/nodes.dmp $(TAXONOMY)/merged.dmp
	$(PYTHON) $(SCRIPTS)/mibig/download_taxonomy.py --blocklist $< --mibig-version 3.1 -o $@ --taxonomy $(TAXONOMY)


# --- Download JGI data ------------------------------------------------------

$(BUILD)/abc/genomes.json:
	mkdir -p $(BUILD)/abc
	$(PYTHON) $(SCRIPTS)/abc/download_genomes.py -o $@

$(BUILD)/abc/clusters.json: $(BUILD)/abc/genomes.json
	$(PYTHON) $(SCRIPTS)/abc/download_clusters.py -i $< -o $@

$(DATA)/datasets/abc/clusters.gbk: $(BUILD)/abc/clusters.json
	$(PYTHON) $(SCRIPTS)/abc/download_records.py -i $< -o $@

$(DATA)/datasets/abc/compounds.json: $(BUILD)/abc/clusters.json $(ATLAS)
	mkdir -p build/cache/abc_compounds
	$(PYTHON) $(SCRIPTS)/abc/download_compounds.py --input $< --output $@ --atlas $(ATLAS) --cache $(BUILD)

# --- Download GenBank data ------------------------------------------------------

$(DATA)/datasets/nuccore/clusters.gbk: $(DATA)/datasets/nuccore/compounds.json
	$(PYTHON) $(SCRIPTS)/nuccore/download_clusters.py --compounds $< --clusters $@

$(DATA)/datasets/nuccore-lite/clusters.gbk: $(DATA)/datasets/nuccore-lite/compounds.json $(DATA)/datasets/nuccore-lite/coordinates.tsv
	$(PYTHON) $(SCRIPTS)/nuccore/download_clusters.py --compounds $< --clusters $@ --coordinates $(word 2,$^)
