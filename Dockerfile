# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.12.8
FROM python:${PYTHON_VERSION}-slim 

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

    
# Set the cache directory to a location within /app
ENV UV_CACHE_DIR=/app/.cache/uv

# Ensure the cache directory is writable
RUN mkdir -p /app/.cache/uv && \
    chown -R appuser:appuser /app/.cache

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:0.5.21 /uv /uvx /bin/

# Install Streamlit and any other dependencies.
RUN pip install streamlit

# Copy the source code into the container.
ADD . .

RUN uv sync --frozen --no-cache

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
#RUN --mount=type=cache,target=/root/.cache/pip \
#    --mount=type=bind,source=requirements.txt,target=requirements.txt \
#    python -m pip install -r requirements.txt

# Install supervisord to manage multiple processes.
RUN apt-get update && apt-get install -y supervisor

# Create a configuration file for supervisord.
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the port that the application listens on.
EXPOSE 8001 8501

# Ensure the log directory is writable
RUN mkdir -p /app && \
    chown -R appuser:appuser /app

# Switch to the non-privileged user to run the application.
USER appuser

# Run supervisord to manage the processes.
CMD ["/usr/bin/supervisord"]
