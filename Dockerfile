# Use Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY mp1/requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire mp1 directory into the container
COPY mp1 /app/mp1

# Hugging Face runs with user 1000 and needs write permissions for certain folders
# We'll ensure corpus and output directories exist and are writable
RUN mkdir -p /app/mp1/corpus /app/mp1/output && \
    chmod -R 777 /app/mp1/corpus /app/mp1/output

# Set the working directory to where main.py and pluto/ are
WORKDIR /app/mp1

# Hugging Face Spaces expose port 7860 by default
EXPOSE 7860

# Command to run the FastAPI app
CMD ["uvicorn", "pluto.server:app", "--host", "0.0.0.0", "--port", "7860"]
