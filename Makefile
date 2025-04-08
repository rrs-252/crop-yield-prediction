init:
    pip install -r requirements.txt

process-data:
    python src/data_processor.py --input data/raw/ICRISAT-District-Level-Data.csv --output data/processed/

train:
    python src/trainer.py

predict:
    python src/main.py --lat 21.1925 --lon 81.2842 --crop rice

test:
    pytest tests/ -v --cov=src --cov-report=html

docker-build:
    docker build -t crop-yield-predictor .

docker-run:
    docker run -it crop-yield-predictor
