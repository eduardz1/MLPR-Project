all:
	python -m project
	svgo -r report/imgs
	typst compile report/report.typ

profile:
	python -m cProfile -m project > profile.txt
