# LaTeX setup
MAIN_TEX := main
TEX_ENGINE := pdflatex
LATEXMK := latexmk
BIBTEX_ENGINE := bibtex
TEX_FLAGS := -shell-escape
LATEXMK_FLAGS := -recorder -shell-escape -interaction=nonstopmode  -synctex=1 #-quiet
PDF_FILE := $(MAIN_TEX).pdf
TEX_COMMAND := $(TEX_ENGINE) $(TEX_FLAGS)

# Docker setup
PWD := $(shell pwd)
DOCKER_IMAGE := srossi/latex:2019
DOCKER_CONTAINER_NAME := latexmk-$(shell basename $(PWD))-$(MAIN_TEX)
DOCKER_RUN_DEAMON := docker run --rm -dt -v $(PWD):$(PWD) --workdir $(PWD) --name $(DOCKER_CONTAINER_NAME) --user="$(shell id -u):$(shell id -g)" $(DOCKER_IMAGE)
DOCKER_RUN := docker run --rm -it -v $(PWD):$(PWD) --workdir $(PWD) --user="$(shell id -u):$(shell id -g)" $(DOCKER_IMAGE)
DOCKER_IT := docker run --rm -it -v $(PWD):$(PWD) --workdir $(PWD)  $(DOCKER_IMAGE)

TEX_FILES := $(shell find . -type f -name "*.tex")
BIB_FILES := $(shell find . -type f -name "*.bib")
TIKZ_FILES := $(shell find . -type f -name "*.tikz")
 
#	Set up viewer and number of parallel jobs for figures
VIEWER :=
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    VIEWER = okular
    NJOBS=$(shell nproc --all)
endif
ifeq ($(UNAME_S),Darwin)
    VIEWER = open -a Skim
    NJOBS=$(shell sysctl -n hw.ncpu)
endif

#	Set up latexmk flags
ifeq ($(TEX_ENGINE),pdflatex)
	LATEXMK_FLAGS += -pdf
else
	LATEXMK_FLAGS += -$(TEX_ENGINE)
endif

# -include $(MAIN_TEX).makefile

pdf: $(PDF_FILE)

docker:
	@$(DOCKER_RUN) $(LATEXMK) $(LATEXMK_FLAGS) -quiet -g $(MAIN_TEX) 


$(PDF_FILE): $(TEX_FILES) $(BIB_FILES) $(TIZK_FILES)
	latexmk $(LATEXMK_FLAGS) -quiet -g $(MAIN_TEX) 


deamon-start:
	@$(DOCKER_RUN_DEAMON) $(LATEXMK) $(LATEXMK_FLAGS) -quiet -pvc -g $(MAIN_TEX) 

deamon-attach: 
	@docker container attach $(DOCKER_CONTAINER_NAME)

deamon-stop:
	@docker container stop $(DOCKER_CONTAINER_NAME)

interactive:
	@$(DOCKER_IT) 

indent:
	@$(DOCKER_RUN)  find . -type f -name "*tex"  -exec latexindent --overwrite -s {} \;

view: 
	$(VIEWER) $(PDF_FILE) &

clean:
	@rm -rf *.sav *.bak *.aux *.bcf *.fdb_latexmk *acn *glo *ist *.fls *.makefile *.run.xml *.bbl *.blg *.figlist *.log  *.out *.toc *.xdv *auxlock *dvi *lof *lol *lot thesis-blx.bib *tdo *acr *alg *dvi *glg *gls *mw *synctex* *.pdf
	find . -name "*.log" -type f -delete
	find . -name "*.dep" -type f -delete
	find . -name "*.dpth" -type f -delete
	find . -name "*.aux" -type f -delete
	find . -name "*bak*" -type f -delete
	find . -name "*.glo*" -type f -delete
	find . -name "*.ist*" -type f -delete
	find . -name "*.acn*" -type f -delete

	
superclean: clean
	@find figures -name "*.pdf" -type f -delete
	@find figures -name "*.md5" -type f -delete

