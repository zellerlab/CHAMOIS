SRC=src
DATA=data
BUILD=build

MIBIG=$(DATA)/mibig
MIBIG_VERSION=3.1

PFAM_VERSION=35.0
PFAM_HMM=$(DATA)/Pfam$(PFAM_VERSION).hmm

ATLAS=$(DATA)/NPAtlas_download.json.gz
CHEMONT=$(DATA)/chemont/ChemOnt_2_1.obo

DATASET_NAMES=abc mibig3.1 mibig2.0
DATASET_TABLES=features classes mibig3.1_ani

PYTHON=python -Wignore
WGET=wget --no-check-certificate

.PHONY: features
features: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/features.hdf5)

.PHONY: classes
classes: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/classes.hdf5)

.PHONY: maccs
maccs: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/maccs.hdf5)

.PHONY: compounds
compounds: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/compounds.json)

.PHONY: clusters
features: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/clusters.gbk)

.PHONY: datasets
datasets: features classes compounds clusters maccs

# --- External data ----------------------------------------------------------

$(PFAM_HMM):
	$(WGET) http://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam$(PFAM_VERSION)/Pfam-A.hmm.gz -O- | gunzip > $@

$(ATLAS):
	$(WGET) https://www.npatlas.org/static/downloads/NPAtlas_download.json -O- | gzip -c > $@

# --- Generic Rules ----------------------------------------------------------

$(DATA)/datasets/%/mibig3.1_ani.hdf5: $(DATA)/datasets/%/clusters.gbk $(DATA)/datasets/mibig3.1/clusters.gbk
	$(PYTHON) src/make_ani.py --query $< --target $(DATA)/datasets/mibig3.1/clusters.gbk -o $@

$(DATA)/datasets/%/features.hdf5: $(DATA)/datasets/%/clusters.gbk $(PFAM_HMM)
	$(PYTHON) src/make_features.py --gbk $< --hmm $(PFAM_HMM) -o $@

$(DATA)/datasets/%/classes.hdf5: $(DATA)/datasets/%/compounds.json $(ATLAS) $(CHEMONT)
	$(PYTHON) src/make_classes.py -i $< -o $@ --atlas $(ATLAS) --chemont $(CHEMONT) --cache $(BUILD)

$(DATA)/datasets/%/maccs.hdf5: $(DATA)/datasets/%/compounds.json $(ATLAS) $(CHEMONT)
	$(PYTHON) src/make_maccs.py -i $< -o $@ --cache $(BUILD)

# --- Download MIBiG 2.0 data ------------------------------------------------

$(DATA)/datasets/mibig2.0/clusters.gbk: $(DATA)/mibig/blocklist.tsv
	$(PYTHON) src/mibig/download_records.py --blocklist $< --mibig-version 2.0 -o $@

$(DATA)/datasets/mibig2.0/compounds.json: $(DATA)/mibig/blocklist.tsv $(ATLAS)
	$(PYTHON) src/mibig/download_compounds.py --blocklist $< --mibig-version 2.0 -o $@ --atlas $(ATLAS) --cache $(BUILD)

$(DATA)/datasets/mibig2.0/taxonomy.tsv: $(DATA)/mibig/blocklist.tsv $(TAXONOMY)/names.dmp $(TAXONOMY)/nodes.dmp $(TAXONOMY)/merged.dmp
	$(PYTHON) src/mibig/download_taxonomy.py --blocklist $< --mibig-version 2.0 -o $@ --taxonomy $(TAXONOMY)

# --- Download MIBiG 3.1 data ------------------------------------------------

$(DATA)/datasets/mibig3.1/clusters.gbk: $(DATA)/mibig/blocklist.tsv
	$(PYTHON) src/mibig/download_records.py --blocklist $< --mibig-version 3.1 -o $@

$(DATA)/datasets/mibig3.1/compounds.json: $(DATA)/mibig/blocklist.tsv $(ATLAS)
	$(PYTHON) src/mibig/download_compounds.py --blocklist $< --mibig-version 3.1 -o $@ --atlas $(ATLAS) --cache $(BUILD)

$(DATA)/datasets/mibig3.1/taxonomy.tsv: $(DATA)/mibig/blocklist.tsv $(TAXONOMY)/names.dmp $(TAXONOMY)/nodes.dmp $(TAXONOMY)/merged.dmp
	$(PYTHON) src/mibig/download_taxonomy.py --blocklist $< --mibig-version 3.1 -o $@ --taxonomy $(TAXONOMY)

# --- Download JGI data ------------------------------------------------------

$(BUILD)/abc/genomes.json:
	mkdir -p $(BUILD)/abc
	$(PYTHON) src/abc/download_genomes.py -o $@

$(BUILD)/abc/clusters.json: $(BUILD)/abc/genomes.json
	$(PYTHON) src/abc/download_clusters.py -i $< -o $@

$(DATA)/datasets/abc/clusters.gbk: $(BUILD)/abc/clusters.json
	$(PYTHON) src/abc/download_records.py -i $< -o $@

$(DATA)/datasets/abc/compounds.json: $(BUILD)/abc/clusters.json $(ATLAS)
	mkdir -p build/cache/abc_compounds
	$(PYTHON) src/abc/download_compounds.py --input $< --output $@ --atlas $(ATLAS) --cache $(BUILD)
