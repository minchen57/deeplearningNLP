FILE=report

LATEX=pdflatex

all:
	latexmk -view=pdf $(FILE)
	#latexmk $(FLAGS) -pvc -view=pdf $(FILE)


pdflatex:
	pdflatex $(FILE)
	biber $(FILE)
	pdflatex $(FILE)
	pdflatex $(FILE)

clean:
	rm -f *.bbl *.log *.blg *.aux *.out *.idx *.run.xml *.bcf *.fdb_latexmk *.fls *.gz *.bak~ *.toc
cleanall: clean
	rm -f *.pdf

view:
	open $(FILE).pdf

