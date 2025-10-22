EMAIL=$(shell git config user.email)

DATA=data
BUILD=build
SCRIPTS=$(DATA)/scripts

GO_VERSION=2025-07-22
GO_OBO=$(DATA)/ontologies/go$(GO_VERSION).obo

MIBIG=$(DATA)/mibig
MIBIG_VERSION=3.1

MITE=$(DATA)/mite
MITE_VERSION=1.18

INTERPRO_VERSION=107.0
INTERPRO_XML=$(DATA)/pfam/interpro$(INTERPRO_VERSION).xml.gz
INTERPRO_JSON=$(DATA)/pfam/interpro$(INTERPRO_VERSION).json

PFAM_VERSION=38.0
PFAM_HMM=$(DATA)/pfam/Pfam$(PFAM_VERSION).hmm

ATLAS=$(DATA)/npatlas/NPAtlas_download.json.gz
CHEMONT=$(DATA)/ontologies/ChemOnt_2_1.obo

DATASET_NAMES=mibig4.0 mibig3.1 mibig2.0 native
DATASET_TABLES=features classes ani

TAXONOMY=$(DATA)/taxonomy
TAXONOMY_VERSION=2025-06-01

PAPER=misc/paper

PYTHON=python -Wignore
WGET=wget --no-check-certificate

# use http://classyfire.wishartlab.com/ to get ClassyFire annotations
WISHART=--wishart

.PHONY: datasets
datasets: features classes compounds clusters

.PHONY: features
features: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/features.hdf5)

.PHONY: classes
classes: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/classes.hdf5)

.PHONY: maccs
maccs: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/maccs.hdf5)

.PHONY: taxonomy
taxonomy: $(foreach dataset,$(DATASET_NAMES),$(DATA)/datasets/$(dataset)/taxonomy.tsv)

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
	$(PYTHON) -m chamois.cli annotate --i $< --hmm $(PFAM_HMM) -o $@

$(DATA)/datasets/%/classification.json: $(DATA)/datasets/%/classes.hdf5
	touch $@

$(DATA)/datasets/%/classes.hdf5: $(DATA)/datasets/%/compounds.json $(ATLAS) $(CHEMONT)
	$(PYTHON) $(SCRIPTS)/common/make_classes.py -i $< -o $@ --atlas $(ATLAS) --chemont $(CHEMONT) --cache $(BUILD) $(WISHART) --output-json $(DATA)/datasets/$*/classification.json

$(DATA)/datasets/%/maccs.hdf5: $(DATA)/datasets/%/compounds.json $(ATLAS) $(CHEMONT)
	$(PYTHON) $(SCRIPTS)/common/make_maccs.py -i $< -o $@

$(DATA)/datasets/%/ani.hdf5: $(DATA)/datasets/%/clusters.gbk $(CHEMONT)
	$(PYTHON) $(SCRIPTS)/common/make_ani.py -q $< -r $< -o $@ -s 0.3

$(DATA)/datasets/%/aci.mibig$(MIBIG_VERSION).hdf5: $(DATA)/datasets/%/clusters.gbk $(DATA)/datasets/mibig$(MIBIG_VERSION)/clusters.gbk
	$(PYTHON) $(SCRIPTS)/common/make_aci.py --query $(word 1,$^) --target $(word 2,$^) -o $@

$(DATA)/datasets/mibig%/types.tsv: $(DATA)/mibig/blocklist.tsv
	$(PYTHON) $(SCRIPTS)/mibig/extract_types.py --mibig-version $* --blocklist data/mibig/blocklist.tsv -o $@

# --- Download MIBiG data ------------------------------------------------------

$(DATA)/datasets/mibig%/clusters.gbk: $(DATA)/mibig/blocklist.tsv $(SCRIPTS)/mibig/download_records.py
	$(PYTHON) $(SCRIPTS)/mibig/download_records.py --blocklist $< --mibig-version $* -o $@ --cache $(BUILD) --email $(EMAIL)

$(DATA)/datasets/mibig%/compounds.json: $(DATA)/mibig/blocklist.tsv $(ATLAS) $(SCRIPTS)/mibig/download_compounds.py
	$(PYTHON) $(SCRIPTS)/mibig/download_compounds.py --blocklist $< --mibig-version $* -o $@ --atlas $(ATLAS) --cache $(BUILD)

$(DATA)/datasets/mibig%/taxonomy.tsv: $(DATA)/mibig/blocklist.tsv $(TAXONOMY)/names.dmp $(TAXONOMY)/nodes.dmp $(TAXONOMY)/merged.dmp
	$(PYTHON) $(SCRIPTS)/mibig/download_taxonomy.py --blocklist $< --mibig-version $* -o $@ --taxonomy $(TAXONOMY)

# --- Download PRISM data ----------------------------------------------------

$(DATA)/prism4/BGCs.tar:
	mkdir -p $(DATA)/datasets/prism4
	$(WGET) https://zenodo.org/records/3985982/files/BGCs.tar?download=1 -O $@

$(DATA)/prism4/predictions.xlsx:
	mkdir -p $(DATA)/datasets/prism4
	$(WGET) https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-19986-1/MediaObjects/41467_2020_19986_MOESM5_ESM.xlsx -O $@

$(DATA)/datasets/prism4/clusters.gbk: $(DATA)/prism4/BGCs.tar $(DATA)/prism4/predictions.xlsx
	$(PYTHON) $(SCRIPTS)/prism4/extract_records.py -i $< -o $@ --table $(word 2,$^)

$(DATA)/datasets/prism4/compounds.json: $(DATA)/prism4/predictions.xlsx $(ATLAS) $(DATA)/datasets/prism4/clusters.gbk
	$(PYTHON) $(SCRIPTS)/prism4/extract_compounds.py -i $< -o $@ --atlas $(word 2,$^) --cache $(BUILD) --clusters $(word 3,$^)


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

$(DATA)/datasets/native/clusters.gbk: $(DATA)/datasets/native/compounds.json $(DATA)/datasets/native/coordinates.tsv
	$(PYTHON) $(SCRIPTS)/native/download_clusters.py --compounds $< --clusters $@ --coordinates $(word 2,$^)

# --- Download MITE data --------------------------------------------------------

$(MITE)/entries.json:
	mkdir -p $(DATA)/mite
	$(PYTHON) $(SCRIPTS)/mite/download_entries.py -o $@

$(MITE)/peptides.json: $(MITE)/entries.json
	$(PYTHON) $(SCRIPTS)/mite/download_peptides.py -i $< -o $@ --email $(EMAIL)

$(MITE)/features.hdf5: $(MITE)/peptides.json $(PFAM_HMM)
	$(PYTHON) $(SCRIPTS)/mite/make_features.py -i $< -o $@ --hmm $(PFAM_HMM)

$(MITE)/classes.hdf5: $(MITE)/entries.json $(CHEMONT) $(ATLAS)
	$(PYTHON) $(SCRIPTS)/mite/make_classes.py -i $< -o $@ --chemont $(CHEMONT) --atlas $(ATLAS)

# --- Train model --------------------------------------------------------------

# path to the trained model
CHAMOIS_WEIGHTS=chamois/predictor/predictor.json
CHAMOIS_HMM=chamois/domains/Pfam$(PFAM_VERSION).hmm.lz4

$(CHAMOIS_WEIGHTS): $(DATA)/datasets/mibig$(MIBIG_VERSION)/classes.hdf5 $(DATA)/datasets/mibig$(MIBIG_VERSION)/features.hdf5
	$(PYTHON) -m chamois.cli train -f $(word 2,$^) -c $(word 1,$^) -o $@
	
$(CHAMOIS_HMM): $(CHAMOIS_WEIGHTS)
	$(PYTHON) setup.py download_pfam -i -f -r

# --- Figures ------------------------------------------------------------------

# Figure 2 - Independent CV
FIG2=$(PAPER)/fig2_cross_validation

$(FIG2)/cv.probas.hdf5: $(FIG2)/cv.report.tsv
	touch $@

$(FIG2)/cv.report.tsv: $(DATA)/datasets/mibig$(MIBIG_VERSION)/features.hdf5 $(DATA)/datasets/mibig$(MIBIG_VERSION)/classes.hdf5
	$(PYTHON) -m chamois.cli cvi -f $(word 1,$^) -c $(word 2,$^) -o $(FIG2)/cv.probas.hdf5 --report $@

$(FIG2)/dummy.probas.hdf5: $(FIG2)/dummy.report.tsv
	touch $@
	
$(FIG2)/dummy.report.tsv: $(DATA)/datasets/mibig$(MIBIG_VERSION)/features.hdf5 $(DATA)/datasets/mibig$(MIBIG_VERSION)/classes.hdf5
	$(PYTHON) -m chamois.cli cvi -f $(word 1,$^) -c $(word 2,$^) -o $(FIG2)/dummy.probas.hdf5 --report $@ --model dummy

$(FIG2)/rf.probas.hdf5: $(FIG2)/rf.report.tsv
	touch $@
	
$(FIG2)/rf.report.tsv: $(DATA)/datasets/mibig$(MIBIG_VERSION)/features.hdf5 $(DATA)/datasets/mibig$(MIBIG_VERSION)/classes.hdf5
	$(PYTHON) -m chamois.cli cvi -f $(word 1,$^) -c $(word 2,$^) -o $(FIG2)/rf.probas.hdf5 --report $@ --model rf

$(FIG2)/cvtree_auprc.html: $(FIG2)/cv.report.tsv
	$(PYTHON) $(FIG2)/tree.py --report $< --output $@

$(FIG2)/pr/.files: $(FIG2)/cv.probas.hdf5 $(DATA)/datasets/mibig$(MIBIG_VERSION)/classes.hdf5
	$(PYTHON) $(FIG2)/prcurves.py --classes $(word 2,$^) --probas $(word 1,$^) -o $(@D)
	touch $@

$(FIG2)/barplot.png: $(DATA)/datasets/mibig$(MIBIG_VERSION)/classes.hdf5 $(DATA)/datasets/mibig$(MIBIG_VERSION)/types.tsv $(FIG2)/cv.probas.hdf5
	$(PYTHON) $(FIG2)/barplot_topk.py --classes $(word 1,$^) --types $(word 2,$^) --probas $(word 3,$^) --output $@

$(FIG2)/barplot.svg: $(DATA)/datasets/mibig$(MIBIG_VERSION)/classes.hdf5 $(DATA)/datasets/mibig$(MIBIG_VERSION)/types.tsv $(FIG2)/cv.probas.hdf5
	$(PYTHON) $(FIG2)/barplot_topk.py --classes $(word 1,$^) --types $(word 2,$^) --probas $(word 3,$^) --output $@

.PHONY: figure2
figure2: $(FIG2)/barplot.svg $(FIG2)/pr/.files $(FIG2)/cvtree_auprc.html

# Figure 4 - Screen Evaluation
FIG4=$(PAPER)/fig4_screen_evaluation

$(FIG4)/predictor.mibig$(MIBIG_VERSION).json: $(DATA)/datasets/mibig$(MIBIG_VERSION)/classes.hdf5 $(DATA)/datasets/mibig$(MIBIG_VERSION)/features.hdf5
	$(PYTHON) -m chamois.cli train -c $(word 1,$^) -f $(word 2,$^) -o $@

$(FIG4)/merged.hdf5: $(FIG4)/predictor.mibig$(MIBIG_VERSION).json 
	$(PYTHON) $(FIG4)/merge_predictions.py

$(FIG4)/dotplot_merged.svg: $(FIG4)/merged.hdf5 $(DATA)/datasets/native/features.hdf5 $(DATA)/datasets/native/classes.hdf5 $(DATA)/datasets/mibig$(MIBIG_VERSION)/features.hdf5 $(DATA)/datasets/mibig$(MIBIG_VERSION)/classes.hdf5
	$(PYTHON) $(FIG4)/dotplot_merged.py

.PHONY: figure4
figure4: $(FIG4)/dotplot_merged.svg

# Supplementary Table 2 - Weights
STBL2=$(PAPER)/sup_table2_weights

$(STBL2)/weights.tsv: $(CHAMOIS_WEIGHTS)
	$(PYTHON) $(STBL2)/extract.py --output $@

.PHONY: suptable2
suptable2:  $(STBL2)/weights.tsv

# Supplementary Table 3 - Unknown domains
STBL3=$(PAPER)/sup_table3_domains

$(STBL3)/table.tsv: $(DATA)/ecdomainminer/EC-Pfam_calculated_associations_Extended.csv $(DATA)/datasets/mibig$(MIBIG_VERSION)/classes.hdf5 $(DATA)/datasets/mibig$(MIBIG_VERSION)/features.hdf5 $(FIG2)/cv.report.tsv $(CHEMONT) $(INTERPRO_XML) $(PFAM_HMM)
	$(PYTHON) $(STBL3)/table.py --chemont $(CHEMONT) --interpro $(INTERPRO_XML) --ec-domain $(word 1,$^) --pfam $(PFAM_HMM) --classes $(word 2,$^) --features $(word 3,$^) --cv-report $(word 4,$^) --output $@

.PHONY: suptable3
suptable3: $(STBL3)/table.tsv
	
# Supplementary Figure 2 - PRISM4 comparison

SFIG2=$(PAPER)/sup_fig2_prism4

$(SFIG2)/probas.hdf5: $(DATA)/datasets/prism4/clusters.gbk $(CHAMOIS_WEIGHTS) $(CHAMOIS_HMM)
	$(PYTHON) -m chamois.cli predict --model $(CHAMOIS_WEIGHTS) -i $< -o $@ --hmm $(CHAMOIS_HMM)

$(SFIG2)/search_results.tsv: $(SFIG2)/probas.hdf5 $(DATA)/npatlas/classes.hdf5 $(CHAMOIS_WEIGHTS)
	$(PYTHON) -m chamois.cli search --model $(CHAMOIS_WEIGHTS) -i $< -c $(word 2,$^) -o $@

$(SFIG2)/boxplot_by_mibig.median_comparison.png: $(SFIG2)/search_results.tsv
	$(PYTHON) $(SFIG2)/plot.py
