# This sets up the container with Python 3.10 installed.
FROM python:3.10.9

# This copies everything in your current directory to the /app directory in the container.
COPY . /LeafAI

# This sets the /app directory as the working directory for any RUN, CMD, ENTRYPOINT, or COPY instructions that follow.
WORKDIR /LeafAI

# Configure pip to be more resilient
RUN pip config set global.timeout 1000 && \
    pip config set global.retries 10

# Install dependencies with increased timeout
RUN pip install --default-timeout=1000 -r requirements.txt

# This tells Docker to listen on port 80 at runtime. Port 80 is the standard port for HTTP.
EXPOSE 80

# Create a streamlit config file
RUN mkdir -p ~/.streamlit/
RUN echo "\
[server]\n\
port = 8501\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > ~/.streamlit/config.toml

# Change the exposed port to 8501 (Streamlit's default port)
EXPOSE 8501

# This sets the default command for the container to run the app with Streamlit.
ENTRYPOINT ["streamlit", "run"]

# This command tells Streamlit to run your app.py script when the container starts.
CMD ["plant_disease_app.py"]
