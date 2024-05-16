all:
	python -m project --all
	typst compile report/report.typ

profile:
	python -m cProfile -m project --all > profile.txt

clean:
	rm report/report.pdf
	rm -r report/imgs/*.svg
	rm profile.txt
