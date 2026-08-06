# Targets run tools as `$(PYTHON) -m ...` so they always use this interpreter,
# whether or not the virtualenv is activated. Override it per invocation:
#
#     make install PYTHON=python3.12
PYTHON ?= $(shell [ -x venv/bin/python ] && echo venv/bin/python || echo python3)

# Defaults shared with scripts/services.sh. Override: make run FRONTEND_PORT=8080
FRONTEND_PORT ?= 8501
API_PORT ?= 8000

# One line: watchfiles takes this as a single quoted argument, and a backslash
# continuation inside the quotes would embed the recipe's indentation in it.
FRONTEND_CMD = $(PYTHON) -m streamlit run nl2sql/ui/app.py --server.port $(FRONTEND_PORT) --server.headless true

.PHONY: help check-python venv install seed inspect test coverage demo eval \
        start stop restart status logs run api clean \
        start-frontend start-backend stop-frontend stop-backend \
        restart-frontend restart-backend logs-frontend logs-backend

help:  ## Show the available targets
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-17s\033[0m %s\n", $$1, $$2}'

check-python:  ## Report the interpreter and pip that the other targets will use
	@$(PYTHON) --version
	@$(PYTHON) -m pip --version

venv:  ## Create the venv/ virtualenv that the other targets will then use
	$(PYTHON) -m venv venv
	@echo "Created venv/ — now run: make install"

install: check-python  ## Install the package and its development dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

seed:  ## Build and populate the demo database
	$(PYTHON) -m nl2sql.cli seed

inspect:  ## Summarise the loaded Knowledge Base
	$(PYTHON) -m nl2sql.cli inspect

test:  ## Run the test suite
	$(PYTHON) -m pytest

coverage:  ## Run the test suite with a coverage report
	$(PYTHON) -m pytest --cov=nl2sql --cov-report=term-missing

demo:  ## Answer the bundled example questions
	$(PYTHON) -m nl2sql.cli demo

eval:  ## Score answer accuracy against the held-out evaluation set
	$(PYTHON) -m nl2sql.cli evaluate

# Detached, so closing the terminal does not stop them. The bare targets act on
# both services; the -frontend and -backend variants act on one.
SERVICES := PYTHON="$(PYTHON)" scripts/services.sh

start:  ## Start the frontend and the backend in the background
	@$(SERVICES) start

stop:  ## Stop both services
	@$(SERVICES) stop

restart:  ## Restart both services, picking up code changes
	@$(SERVICES) restart

status:  ## Show whether each service is running, and on which port
	@$(SERVICES) status

logs:  ## Follow live output from both services (Ctrl-C to stop)
	@$(SERVICES) logs -f

start-frontend:  ## Start only the web interface
	@$(SERVICES) start frontend

start-backend:  ## Start only the HTTP API
	@$(SERVICES) start backend

stop-frontend:  ## Stop only the web interface
	@$(SERVICES) stop frontend

stop-backend:  ## Stop only the HTTP API
	@$(SERVICES) stop backend

restart-frontend:  ## Restart only the web interface
	@$(SERVICES) restart frontend

restart-backend:  ## Restart only the HTTP API
	@$(SERVICES) restart backend

logs-frontend:  ## Follow live output from the web interface
	@$(SERVICES) logs frontend -f

logs-backend:  ## Follow live output from the HTTP API
	@$(SERVICES) logs backend -f

# The web interface builds the pipeline in-process and never calls the HTTP API, so
# `run` alone is the whole system. `api` is a second, independent way in.
run:  ## Run the web interface in the foreground, reloading on change
	@echo "Web interface: http://127.0.0.1:$(FRONTEND_PORT)"
	@$(PYTHON) -m watchfiles --filter python --sigint-timeout 2 "$(FRONTEND_CMD)" nl2sql

api:  ## Run the HTTP API in the foreground, reloading on change
	@echo "API docs: http://127.0.0.1:$(API_PORT)/docs"
	@$(PYTHON) -m uvicorn nl2sql.api:app --port $(API_PORT) --reload --reload-dir nl2sql

clean: stop  ## Stop the services, then remove build artefacts, caches and the demo database
	rm -rf build dist .pytest_cache .ruff_cache .coverage htmlcov *.egg-info \
		data/observability.db .run logs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
