all:
	python -m project
	svgo -r report/imgs
	typst compile report/report.typ
