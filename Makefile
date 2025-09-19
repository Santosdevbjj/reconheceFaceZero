# Makefile para gerenciar o projeto reconheceFaceZero com Docker Compose

PROJECT_NAME = reconhece-face-gpu

# ===========================
# Comandos principais
# ===========================

## Build do container
build:
	docker-compose build

## Sobe o container em foreground
up:
	docker-compose up

## Sobe o container em background (detached mode)
up-detach:
	docker-compose up -d

## Para e remove os containers
down:
	docker-compose down

## Reinicia o container
restart: down up-detach

## Acessa o shell do container
shell:
	docker exec -it $(PROJECT_NAME) bash

## Ver logs do container
logs:
	docker-compose logs -f

## Remove imagens não utilizadas
clean:
	docker system prune -f
	docker volume prune -f
	docker image prune -f

# ===========================
# Atalhos para Jupyter
# ===========================

## Mostra a URL do Jupyter no container
jupyter-token:
	docker logs $(PROJECT_NAME) 2>&1 | grep -m 1 "http://127.0.0.1:8888/"

## Abre o Jupyter no navegador
open-jupyter:
	xdg-open http://127.0.0.1:8888 || open http://127.0.0.1:8888

# ===========================
# Help
# ===========================

help:
	@echo "Comandos disponíveis:"
	@echo "  make build        -> Build do container"
	@echo "  make up           -> Sobe o container em foreground"
	@echo "  make up-detach    -> Sobe o container em background"
	@echo "  make down         -> Para e remove os containers"
	@echo "  make restart      -> Reinicia o container"
	@echo "  make shell        -> Acessa o shell do container"
	@echo "  make logs         -> Mostra logs do container"
	@echo "  make clean        -> Limpa imagens/volumes não utilizados"
	@echo "  make jupyter-token-> Mostra a URL/token do Jupyter"
	@echo "  make open-jupyter -> Abre o Jupyter no navegador"
