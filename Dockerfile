ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}

# https://python-poetry.org/docs#ci-recommendations
ENV POETRY_VERSION=1.7.0
# ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv

# Tell Poetry where to place its cache and virtual environment
ENV POETRY_CACHE_DIR=/opt/.cache

# Creating a virtual environment just for poetry and install it with pip
RUN python3 -m venv $POETRY_VENV \
	&& $POETRY_VENV/bin/pip install -U pip setuptools \
	&& $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}


# Add Poetry to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

ENV POETRY_VIRTUALENVS_IN_PROJECT=true

CMD ["bash"]
