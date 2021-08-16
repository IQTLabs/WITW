SHELL:=/bin/bash
PIP=$(shell which pip3 || echo "pip3")


build:
	@docker build -t witw-model .
train_cvusa: build train_cvusa_nobuild
train_cvusa_nobuild:
	@echo
	@echo "Running Train on data files $(DATA) with fov $(FOV)"
	@docker docker run -it -v "$(DATA):/data" -v "$(PWD)/weights:/weights" --ipc=host --gpus all witw-model python3 cvig_fov.py --mode train --dataset cvusa --fov $(FOV)
	@echo
train_witw: build train_cvusa_nobuild
train_witw_nobuild:
	@echo
	@echo "Running Train on data files $(DATA) with fov $(FOV)"
	@docker docker run -it -v "$(DATA):/data2" -v "$(PWD)/weights:/weights" --ipc=host --gpus all witw-model python3 cvig_fov.py --mode train --dataset witw --fov $(FOV)
	@echo
test_cvusa: build test_cvusa_nobuild
test_cvusa_nobuild:
	@echo
	@echo "Running Train on data files $(DATA) with fov $(FOV)"
	@docker docker run -it -v "$(DATA):/data" -v "$(PWD)/weights:/weights" --ipc=host --gpus all witw-model python3 cvig_fov.py --mode test --dataset cvusa --fov $(FOV)
	@echo
test_witw: build train_cvusa_nobuild
test_witw_nobuild:
	@echo
	@echo "Running Train on data files $(DATA) with fov $(FOV)"
	@docker docker run -it -v "$(DATA):/data2" -v "$(PWD)/weights:/weights" --ipc=host --gpus all witw-model python3 cvig_fov.py --mode test --dataset witw --fov $(FOV)
	@echo
