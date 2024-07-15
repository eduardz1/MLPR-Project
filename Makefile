all:
	@python -m project -a -c

profile:
	@python -m cProfile -s tottime -m project -a > profile.txt

clean:
	@find . -name "*.svg" -type f -delete
	@find . -name "*.pdf" -type f -delete
	@find . -name "*.npy" -type f -delete
	@find . -name "profile*" -type f -delete
	@find . -name "*.pyc" -type f -delete
	@find . -name "*.nbc" -type f -delete
	@find . -name "*.nbi" -type f -delete
	@find . -name "_*.json" -type f -delete
