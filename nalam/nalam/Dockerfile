# Use the official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (for cache efficiency)
COPY requirements.txt .

# Install dependencies
# We allow creating a cache to speed up builds
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# This includes app.py, the helper python files, AND the 'nalam_chroma_db' folder
COPY . .

# Create a user to run the app (Hugging Face Spaces requirement for security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose the port (Hugging Face default is 7860)
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]