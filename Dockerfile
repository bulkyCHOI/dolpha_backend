FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

# CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]