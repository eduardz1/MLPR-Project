all:
	@python -m project -a -c

profile:
	@python -m cProfile -m project -a > profile.txt

clean:
	@find . -name "*.svg" -type f -delete
	@find . -name "*.pdf" -type f -delete
