SRC=src
DATA=data
BUILD=build

MIBIG=$(DATA)/mibig
MIBIG_VERSION=3.1

PFAM_VERSION=35.0
PFAM_HMM=$(DATA)/Pfam$(PFAM_VERSION).hmm

ATLAS=$(DATA)/NPAtlas_download.json.gz
CHEMONT=$(DATA)/chemont/ChemOnt_2_1.obo

DATASET_NAMES=abc mibig
DATASET_TABLES=features mibig_ani

PYTHON=python -Wignore

.PHONY: datasets
datasets: $(foreach dataset,$(DATASET_NAMES),$(foreach table,$(DATASET_TABLES),$(DATA)/datasets/$(dataset)/$(table).hdf5))

# --- Download or prepare data -----------------------------------------------

$(PFAM_HMM):
	wget http://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam$(PFAM_VERSION)/Pfam-A.hmm.gz -O- | gunzip > $@

$(ATLAS):
	wget https://www.npatlas.org/static/downloads/NPAtlas_download.json -O- | gzip -c > $@

# --- Patch / map MIBiG compounds to PubChem or NPAtlas ----------------------

$(BUILD)/mibig-mapped.json: $(MIBIG)/mibig_json_$(MIBIG_VERSION).tar.gz $(ATLAS)
	$(PYTHON) $(SRC)/map_compounds.py --json $< -o $@ --atlas $(ATLAS)

# --- Get ClassyFire classes for every compound ------------------------------

$(BUILD)/mibig-classified.json: $(BUILD)/mibig-mapped.json $(ATLAS)
	$(PYTHON) $(SRC)/classify_compounds.py -i $< --atlas $(ATLAS) -o $@

# --- Generic Rules ----------------------------------------------------------

$(DATA)/datasets/%/mibig_ani.hdf5: $(DATA)/datasets/%/clusters.gbk $(DATA)/datasets/mibig/clusters.gbk
	$(PYTHON) src/make_ani.py --query $< --target $(DATA)/datasets/mibig/clusters.gbk -o $@

$(DATA)/datasets/%/features.hdf5: $(DATA)/datasets/%/clusters.gbk $(PFAM_HMM)
	$(PYTHON) src/make_features.py --gbk $< --hmm $(PFAM_HMM) -o $@

$(DATA)/datasets/%/classes.hdf5: $(DATA)/datasets/%/compounds.json $(ATLAS) $(CHEMONT)
	$(PYTHON) src/make_classes.py -i $< -o $@ --atlas $(ATLAS) --chemont $(CHEMONT)

# --- Download MIBiG data ----------------------------------------------------

$(DATA)/datasets/mibig/clusters.gbk: $(DATA)/mibig/blocklist.tsv
	$(PYTHON) src/mibig/download_records.py --blocklist $< --mibig-version $(MIBIG_VERSION) -o $@

$(DATA)/datasets/mibig/compounds.json: $(DATA)/mibig/blocklist.tsv $(ATLAS)
	$(PYTHON) src/mibig/download_compounds.py --blocklist $< --mibig-version $(MIBIG_VERSION) -o $@ --atlas $(ATLAS)

# --- Download JGI data ------------------------------------------------------

$(BUILD)/abc/genomes.json:
	mkdir -p $(BUILD)/abc
	$(PYTHON) src/abc/download_genomes.py -o $@

$(BUILD)/abc/clusters.json: $(BUILD)/abc/genomes.json
	$(PYTHON) src/abc/download_clusters.py -i $< -o $@

$(DATA)/datasets/abc/clusters.gbk: $(BUILD)/abc/clusters.json
	$(PYTHON) src/abc/download_records.py -i $< -o $@

$(DATA)/datasets/abc/compounds.json: $(BUILD)/abc/clusters.json
	$(PYTHON) src/abc/download_compounds.py --input $< --output $@
