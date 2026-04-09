FROM python:3.11-slim

# Create a non-root user that Hugging Face Spaces requires
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
	TF_USE_LEGACY_KERAS=1

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

# Run the FastAPI server on port 7860 standard for HF Spaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
