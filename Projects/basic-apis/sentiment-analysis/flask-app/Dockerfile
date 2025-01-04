FROM python:3.13.1-slim

WORKDIR /app

COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY app.py .

CMD ["python", "app.py"]