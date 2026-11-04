.PHONY: install lint test run smoke docker docker-run all

install:      ## app + every provider SDK + dev tools
	pip install -e ".[dev]"

lint:         ## ruff + mypy
	ruff check src tests && ruff format --check src tests && mypy src

test:         ## offline suite: registry, service (swap, fallback, trimming, streaming), API, provider adapters
	pytest -q

run:          ## API on 127.0.0.1:8080 with the offline model as default
	convo serve

smoke:        ## end-to-end against a running server
	CONVO_URL=$${URL:-http://127.0.0.1:8080} ./scripts/convo.sh models
	CONVO_URL=$${URL:-http://127.0.0.1:8080} ./scripts/convo.sh ask "smoke test" offline

docker:
	docker build -t conversation-api .

docker-run:   ## pass provider credentials through as needed, e.g. -e AWS_REGION -e AWS_PROFILE
	docker run --rm -p 8080:8080 -e CONVO_API_KEY conversation-api

all: lint test
