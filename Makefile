SRC=src
DATA=data
BUILD=build

MIBIG=$(DATA)/mibig
MIBIG_VERSION=3.1

PFAM_VERSION=35.0
PFAM_HMM=$(DATA)/Pfam$(PFAM_VERSION).hmm.gz

# --- Download or prepare data -----------------------------------------------

$(MIBIG)/mibig_gbk_$(MIBIG_VERSION).tar.gz:
	mkdir -p $(MIBIG)
	wget https://dl.secondarymetabolites.org/mibig/mibig_gbk_$(MIBIG_VERSION).tar.gz -O $@

$(MIBIG)/mibig_json_$(MIBIG_VERSION).tar.gz:
	mkdir -p $(MIBIG)
	wget https://dl.secondarymetabolites.org/mibig/mibig_json_$(MIBIG_VERSION).tar.gz -O $@

$(PFAM_HMM):
	wget http://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam$(PFAM_VERSION)/Pfam-A.hmm.gz -O $@

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