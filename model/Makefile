SHELL:=/bin/bash
PIP=$(shell which pip3 || echo "pip3")


build:
	@docker build -t witw-model .
train_cvusa: build train_cvusa_nobuild
train_cvusa_nobuild:
	@echo
	@echo "Running Train on data files $(DATA) with fov $(FOV)"
	@docker run -it -v "$(DATA):/witw-model/data" -v "$(PWD)/weights:/witw-model/weights" --ipc=host --gpus all witw-model --mode train --dataset cvusa --fov $(FOV)
	@echo
train_witw: build train_witw_nobuild
train_witw_nobuild:
	@echo
	@echo "Running Train on data files $(DATA) with fov $(FOV)"
	@docker run -it -v "$(DATA):/witw-model/data2" -v "$(PWD)/weights:/witw-model/weights" --ipc=host --gpus all witw-model --mode train --dataset witw --fov $(FOV)
	@echo
test_cvusa: build test_cvusa_nobuild
test_cvusa_nobuild:
	@echo
	@echo "Running Train on data files $(DATA) with fov $(FOV)"
	@docker run -it -v "$(DATA):/witw-model/data" -v "$(PWD)/weights:/witw-model/weights" --ipc=host --gpus all witw-model --mode test --dataset cvusa --fov $(FOV)
	@echo
test_witw: build test_witw_nobuild
test_witw_nobuild:
	@echo
	@echo "Running Train on data files $(DATA) with fov $(FOV)"
	@docker run -it -v "$(DATA):/data2" -v "$(PWD)/weights:/weights" --ipc=host --gpus all witw-model --mode test --dataset witw --fov $(FOV)
	@echo
