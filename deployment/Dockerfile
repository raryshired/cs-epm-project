# Dockerfile for Hugging Face Spaces - Streamlit deployment
FROM python:3.11-slim

# Create non-root user (uid 1000 for HF Spaces compliance)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user . .

# Expose Streamlit default port
EXPOSE 8501

# Streamlit entrypoint
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
