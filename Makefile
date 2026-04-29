test: data_test.py
	@echo Testing data.py ...
	@chmod +x ./data_test.py
	./data_test.py -v
	@echo

demo:
	@echo Running demo ...
	@chmod +x main.py
	./main.py
	@echo