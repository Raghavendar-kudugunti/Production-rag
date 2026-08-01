FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV KMP_DUPLICATE_LIB_OK=TRUE

EXPOSE 10000

CMD ["chainlit", "run", "app/chainlit_app.py", "--host", "0.0.0.0", "--port", "10000"]