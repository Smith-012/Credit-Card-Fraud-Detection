FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
CMD ["python", "main.py", "--data", "data/creditcard.csv", "--out", "artifacts"]
