FROM python:3.12
RUN apt update
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000

CMD ["python", "-m", "api.main"]