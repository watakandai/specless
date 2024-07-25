ARG PYTHON_VERSION=3.8
FROM python:${PYTHON_VERSION}

RUN apt-get update && \
	apt install -y graphviz

# https://python-poetry.org/docs#ci-recommendations
ENV POETRY_VERSION=1.7.0 \
	# Poetry home directory
    POETRY_HOME='/usr/local' \
	# Add Poetry's bin folder to the PATH
	PATH="/usr/local/bin:$PATH" \
	# Avoids any interactions with the terminal
    POETRY_NO_INTERACTION=1 \
	# This avoids poetry from creating a virtualenv
	# Instead, it directly installs the dependencies in the system's python environment
    POETRY_VIRTUALENVS_CREATE=false

# System deps:
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy the project files
WORKDIR /home/specless
COPY pyproject.toml poetry.lock /home/specless/

# Project initialization and conditionally install cvxopt if on x86 architecture
RUN poetry install --no-interaction
# RUN poetry install --no-interaction && \
#     if [ "$(uname -m)" = "x86_64" ]; then poetry add cvxopt; fi

CMD ["bash"]
