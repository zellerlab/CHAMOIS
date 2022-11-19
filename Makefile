SRC=src
DATA=data
BUILD=build

MIBIG=$(DATA)/mibig
MIBIG_VERSION=3.1

PFAM_VERSION=35.0
PFAM_HMM=$(DATA)/Pfam$(PFAM_VERSION).hmm

DATASET_NAMES=abc mibig
DATASET_TABLES=features mibig_ani

.PHONY: datasets
datasets: $(foreach dataset,$(DATASET_NAMES),$(foreach table,$(DATASET_TABLES),$(DATA)/datasets/$(dataset)/$(table).hdf5))

# --- Download or prepare data -----------------------------------------------

$(PFAM_HMM):
	wget http://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam$(PFAM_VERSION)/Pfam-A.hmm.gz -O- | gunzip > $@

$(DATA)/NPAtlas_download.json.gz:
	wget https://www.npatlas.org/static/downloads/NPAtlas_download.json -O- | gzip -c > $@

# --- Patch / map MIBiG compounds to PubChem or NPAtlas ----------------------

$(BUILD)/mibig-mapped.json: $(MIBIG)/mibig_json_$(MIBIG_VERSION).tar.gz $(DATA)/NPAtlas_download.json.gz
	python $(SRC)/map_compounds.py --json $< -o $@ --atlas $(DATA)/NPAtlas_download.json.gz

# --- Get ClassyFire classes for every compound ------------------------------

$(BUILD)/mibig-classified.json: $(BUILD)/mibig-mapped.json $(DATA)/NPAtlas_download.json.gz
	python $(SRC)/classify_compounds.py -i $< --atlas $(DATA)/NPAtlas_download.json.gz -o $@

# --- Generic Rules ----------------------------------------------------------

$(DATA)/datasets/%/mibig_ani.hdf5: $(DATA)/datasets/%/clusters.gbk $(DATA)/datasets/mibig/clusters.gbk
	python src/make_ani.py --query $< --target $(DATA)/datasets/mibig/clusters.gbk -o $@

$(DATA)/datasets/%/features.hdf5: $(DATA)/datasets/%/clusters.gbk $(PFAM_HMM)
	python src/make_features.py --gbk $< --hmm $(PFAM_HMM) -o $@

# --- Download MIBiG data ----------------------------------------------------

$(DATA)/datasets/mibig/clusters.gbk:
	mkdir -p $(BUILD)/mibig
	wget "https://dl.secondarymetabolites.org/mibig/mibig_gbk_3.1.tar.gz" -O- | tar xzO > $@

# --- Download JGI data ------------------------------------------------------

$(BUILD)/abc/genomes.json:
	mkdir -p $(BUILD)/abc
	python src/abc/download_genomes.py -o $@

$(BUILD)/abc/clusters.json: $(BUILD)/abc/genomes.json
	mkdir -p $(BUILD)/abc
	python src/abc/download_clusters.py -i $< -o $@

$(DATA)/datasets/abc/clusters.gbk: $(BUILD)/abc/clusters.json
	mkdir -p $(BUILD)/abc
	python src/abc/download_records.py -i $< -o $@
