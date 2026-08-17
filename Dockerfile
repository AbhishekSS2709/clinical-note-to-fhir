FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml .
RUN pip3 install --no-cache-dir -e ".[gpu,serve]"
COPY src/ src/
COPY configs/ configs/
COPY web/ web/
EXPOSE 8000
CMD ["uvicorn", "fhir_extract.serve:app", "--host", "0.0.0.0", "--port", "8000"]
