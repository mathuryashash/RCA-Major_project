FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies (curl for healthcheck, build-essential for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Add a healthcheck for container orchestration (like Kubernetes or AWS ECS)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start the Streamlit application
ENTRYPOINT ["streamlit", "run", "src/reporting/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
