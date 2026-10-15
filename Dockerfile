# CPU image for running the tests, the smoke test and the results scripts.
# For GPU training, install a CUDA build of PyTorch that matches your driver instead of the default wheel.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Default: run the unit tests and the synthetic smoke test.
CMD ["pytest", "-q"]
