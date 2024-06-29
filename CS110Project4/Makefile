CC = gcc
CFLAGS = -Wall -Werror -O2 -fno-tree-vectorize -fno-tree-loop-vectorize -fno-tree-slp-vectorize -fopenmp -march=native
MEMCHECK = valgrind --tool=memcheck --leak-check=full --track-origins=yes

BUILD_DIR = ./build
BIN = $(BUILD_DIR)/test

$(shell mkdir -p $(BUILD_DIR))

$(BIN): src/test.c src/baseline.c src/impl.c src/impl_experiment.c src/impl_old.c
	@$(CC) $(CFLAGS) $^ -o $@

run: $(BIN)	
	@$^

memcheck: $(BIN)
	$(MEMCHECK) $(BIN)

clean: 
	-@rm -rf $(BUILD_DIR)

.PHONY: run clean memcheck
