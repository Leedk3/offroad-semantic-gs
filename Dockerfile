FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.txt ./
COPY src ./src
COPY licenses ./licenses
RUN python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir .
ENTRYPOINT ["offroad-gs"]
CMD ["--help"]
