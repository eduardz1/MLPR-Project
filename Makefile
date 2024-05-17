all:
	python -m project -a -c

profile:
	python -m cProfile -m project -a > profile.txt

clean:
	rm -r */*.pdf */*.svg profile.txt
