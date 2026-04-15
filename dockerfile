FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget curl bzip2 ca-certificates git tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p $CONDA_DIR \
    && rm /tmp/miniforge.sh \
    && conda clean --all --yes

RUN conda create -n ragenv python=3.11 -y

SHELL ["conda", "run", "-n", "ragenv", "/bin/bash", "-c"]

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN python -c "import streamlit; print('Streamlit OK')" \
    && python -c "from langchain_community.embeddings import HuggingFaceEmbeddings; print('Hugging Face embeddings OK')" \
    && python -c "from docx import Document; print('python-docx OK')"

COPY . .

RUN mkdir -p /app/data /app/outputs /root/.streamlit \
    && printf '[server]\nport = 8501\naddress = "0.0.0.0"\nheadless = true\nmaxUploadSize = 1024\n\n[browser]\ngatherUsageStats = false\n' > /root/.streamlit/config.toml

EXPOSE 8501

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ragenv", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
