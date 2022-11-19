SRC=src
DATA=data
BUILD=build

MIBIG=$(DATA)/mibig
MIBIG_VERSION=3.1

PFAM_VERSION=35.0
PFAM_HMM=$(DATA)/Pfam$(PFAM_VERSION).hmm

DATASET_NAMES=abc
DATASET_TABLES=features classes mibig_ani

.PHONY: datasets
datasets: $(foreach dataset,$(DATASET_NAMES),$(foreach table,$(DATASET_TABLES),$(DATA)/datasets/$(dataset)/$(table).hdf5))

# --- Download or prepare data -----------------------------------------------

$(MIBIG)/mibig_gbk_$(MIBIG_VERSION).tar.gz:
	mkdir -p $(MIBIG)
	wget https://dl.secondarymetabolites.org/mibig/mibig_gbk_$(MIBIG_VERSION).tar.gz -O $@

$(MIBIG)/mibig_json_$(MIBIG_VERSION).tar.gz:
	mkdir -p $(MIBIG)
	wget https://dl.secondarymetabolites.org/mibig/mibig_json_$(MIBIG_VERSION).tar.gz -O $@

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


# --- Compute domain compositions --------------------------------------------

$(BUILD)/compositions/Pfam$(PFAM_VERSION)/counts.npz $(BUILD)/compositions/Pfam$(PFAM_VERSION)/compositions.npz $(BUILD)/compositions/Pfam$(PFAM_VERSION)/domains.tsv $(BUILD)/compositions/Pfam$(PFAM_VERSION)/labels.tsv: $(MIBIG)/mibig_gbk_$(MIBIG_VERSION).tar.gz $(MIBIG)/mibig_json_$(MIBIG_VERSION).tar.gz $(PFAM_HMM)
	python src/make_compositions.py -o $(BUILD)/compositions/Pfam$(PFAM_VERSION)/ --gbk $(MIBIG)/mibig_gbk_$(MIBIG_VERSION).tar.gz --json $(MIBIG)/mibig_json_$(MIBIG_VERSION).tar.gz --hmm $(PFAM_HMM)


# --- Compute ANI between clusters -------------------------------------------

$(MIBIG)/ani_$(MIBIG_VERSION).coo.npz: $(MIBIG)/mibig_gbk_$(MIBIG_VERSION).tar.gz
	python src/compute_ani_matrix.py --gbk $< -o $@


# --- Evaluate predictors ----------------------------------------------------

$(BUILD)/classifier-stats.tsv $(BUILD)/classifier-curves.json: $(DATA)/Pfam$(PFAM_VERSION).txt $(BUILD)/compositions/Pfam$(PFAM_VERSION)/counts.npz $(BUILD)/compositions/Pfam$(PFAM_VERSION)/compositions.npz $(BUILD)/compositions/Pfam$(PFAM_VERSION)/domains.tsv $(BUILD)/compositions/Pfam$(PFAM_VERSION)/labels.tsv $(DATA)/chemont/ChemOnt_2_1.obo $(BUILD)/mibig-classified.json
	# FIXME: remove hardcoded paths
	python src/build_predictors.py 



# --- Generic Rules ----------------------------------------------------------

$(DATA)/datasets/%/mibig_ani.hdf5: $(BUILD)/%/clusters.gbk $(BUILD)/mibig/clusters.gbk
	python src/make_ani.py --query $< --target $(BUILD)/mibig/clusters.gbk -o $@




# === MIBIG ==================================================================

# --- Download data ----------------------------------------------------------

$(BUILD)/mibig/clusters.gbk:
	mkdir -p $(BUILD)/mibig
	wget "https://dl.secondarymetabolites.org/mibig/mibig_gbk_3.1.tar.gz" -O- | tar xzO > $@


# === JGI ABC ================================================================

# --- Download metadata ------------------------------------------------------

$(BUILD)/abc/genomes.json:
	mkdir -p $(BUILD)/abc
	python src/abc/download_genomes.py -o $@

$(BUILD)/abc/clusters.json: $(BUILD)/abc/genomes.json
	mkdir -p $(BUILD)/abc
	python src/abc/download_clusters.py -i $< -o $@

$(BUILD)/abc/clusters.gbk: $(BUILD)/abc/clusters.json
	mkdir -p $(BUILD)/abc
	python src/abc/download_records.py -i $< -o $@

$(DATA)/datasets/abc/features.hdf5: $(BUILD)/abc/clusters.gbk $(PFAM_HMM)
	mkdir -p $(DATA)/datasets/abc
	python src/make_features.py --gbk $< --hmm $(PFAM_HMM) -o $@



