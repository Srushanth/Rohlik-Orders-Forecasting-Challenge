# Create virtual environment using conda
venv-conda:
	conda create -p .rohlik python==3.11 -y

install:
	pip install -r requirements.txt

lint:
	python -m pylint *.py