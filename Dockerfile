FROM python:3.10
RUN apt update
WORKDIR /app
COPY models/model.onnx /app/models/
COPY models/scaler.pkl /app/models/
COPY api /app/api/
COPY requirements.txt /app/
RUN pwd && ls
RUN pip install -r requirements.txt
EXPOSE 5000

CMD ["python", "-m", "api.main"]