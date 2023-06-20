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

PFAM_VERSION=35.0
PFAM_HMM=$(DATA)/pfam/Pfam$(PFAM_VERSION).hmm

PGAP_VERSION=11.0
PGAP_HMM=$(DATA)/PGAP$(PGAP_VERSION).hmm

KOFAM_DATE=2023-01-01
KOFAM_LIST=$(DATA)/Kofam$(KOFAM_DATE).tsv
KOFAM_HMM=$(DATA)/Kofam$(KOFAM_DATE).hmm

SMCOGS_VERSION=6-1-1
SMCOGS_HMM=$(DATA)/smCOGS$(SMCOGS_VERSION).hmm

ATLAS=$(DATA)/npatlas/NPAtlas_download.json.gz
CHEMONT=$(DATA)/ontologies/ChemOnt_2_1.obo

DATASET_NAMES=mibig3.1 mibig2.0
DATASET_TABLES=pfam35 classes mibig3.1_ani

PYTHON=python -Wignore
WGET=wget --no-check-certificate

.PHONY: datasets
datasets: features classes compounds clusters maccs

.PHONY: pfam35
pfam35: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/pfam35.hdf5)

.PHONY: kofam2023
kofam2023: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/kofam2023.hdf5)

.PHONY: pgap11
pgap11: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/pgap11.hdf5)

.PHONY: smcogs6
smcogs6: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/smcogs6.hdf5)

.PHONY: features
features: pfam35 # kofam2023 pgap11

.PHONY: classes
classes: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/classes.hdf5)

.PHONY: maccs
maccs: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/maccs.hdf5)

.PHONY: compounds
compounds: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/compounds.json)

.PHONY: clusters
clusters: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/clusters.gbk)


# --- External data ----------------------------------------------------------

$(PFAM_HMM):
	$(WGET) http://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam$(PFAM_VERSION)/Pfam-A.hmm.gz -O- | gunzip > $@

$(KOFAM_HMM): $(KOFAM_LIST)
	$(WGET) ftp://ftp.genome.jp/pub/db/kofam/archives/$(KOFAM_DATE)/profiles.tar.gz -O- | tar xz --wildcards '*.hmm' -O > $@
	$(PYTHON) $(SCRIPTS)/kofam/set_thresholds.py --hmm $@ --list $<

$(KOFAM_LIST):
	$(WGET) ftp://ftp.genome.jp/pub/db/kofam/archives/$(KOFAM_DATE)/ko_list.gz -O | gunzip -c > $@

$(PGAP_HMM):
	$(WGET) ftp://ftp.ncbi.nlm.nih.gov/hmm/11.0/hmm_PGAP.LIB -O $@

$(SMCOGS_HMM):
	$(PYTHON) $(SCRIPTS)/smcogs/download.py --version $(SMCOGS_VERSION) --output $@

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

$(DATA)/datasets/%/mibig3.1_ani.hdf5: $(DATA)/datasets/%/clusters.gbk $(DATA)/datasets/mibig3.1/clusters.gbk
	$(PYTHON) $(SCRIPTS)/common/make_ani.py --query $< --target $(DATA)/datasets/mibig3.1/clusters.gbk -o $@

$(DATA)/datasets/%/pfam35.hdf5: $(DATA)/datasets/%/clusters.gbk $(PFAM_HMM)
	$(PYTHON) -m conch.cli annotate --i $< --hmm $(PFAM_HMM) -o $@

$(DATA)/datasets/%/kofam2023.hdf5: $(DATA)/datasets/%/clusters.gbk $(KOFAM_HMM)
	$(PYTHON) -m conch.cli annotate --gbk $< --hmm $(KOFAM_HMM) -o $@

$(DATA)/datasets/%/pgap11.hdf5: $(DATA)/datasets/%/clusters.gbk $(PGAP_HMM)
	$(PYTHON) -m conch.cli annotate --gbk $< --hmm $(PGAP_HMM) -o $@

$(DATA)/datasets/%/smcogs6.hdf5: $(DATA)/datasets/%/clusters.gbk $(SMCOGS_HMM)
	$(PYTHON) -m conch.cli annotate --gbk $< --hmm $(SMCOGS_HMM) -o $@

$(DATA)/datasets/%/classes.hdf5: $(DATA)/datasets/%/compounds.json $(ATLAS) $(CHEMONT)
	$(PYTHON) $(SCRIPTS)/common/make_classes.py -i $< -o $@ --atlas $(ATLAS) --chemont $(CHEMONT) --cache $(BUILD)

$(DATA)/datasets/%/maccs.hdf5: $(DATA)/datasets/%/compounds.json $(ATLAS) $(CHEMONT)
	$(PYTHON) $(SCRIPTS)/common/make_maccs.py -i $< -o $@

# --- Download MIBiG 2.0 data ------------------------------------------------

$(DATA)/datasets/mibig2.0/clusters.gbk: $(DATA)/mibig/blocklist.tsv
	$(PYTHON) $(SCRIPTS)/mibig/download_records.py --blocklist $< --mibig-version 2.0 -o $@

$(DATA)/datasets/mibig2.0/compounds.json: $(DATA)/mibig/blocklist.tsv $(ATLAS)
	$(PYTHON) $(SCRIPTS)/mibig/download_compounds.py --blocklist $< --mibig-version 2.0 -o $@ --atlas $(ATLAS) --cache $(BUILD)

$(DATA)/datasets/mibig2.0/taxonomy.tsv: $(DATA)/mibig/blocklist.tsv $(TAXONOMY)/names.dmp $(TAXONOMY)/nodes.dmp $(TAXONOMY)/merged.dmp
	$(PYTHON) $(SCRIPTS)/mibig/download_taxonomy.py --blocklist $< --mibig-version 2.0 -o $@ --taxonomy $(TAXONOMY)

# --- Download MIBiG 3.1 data ------------------------------------------------

$(DATA)/datasets/mibig3.1/clusters.gbk: $(DATA)/mibig/blocklist.tsv
	$(PYTHON) $(SCRIPTS)/mibig/download_records.py --blocklist $< --mibig-version 3.1 -o $@

$(DATA)/datasets/mibig3.1/compounds.json: $(DATA)/mibig/blocklist.tsv $(ATLAS)
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

# --- Download Nuccore data --------------------------------------------------

