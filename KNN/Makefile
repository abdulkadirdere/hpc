INC="./inc"
FLAGS=-I$(INC)
OMPFLAG=-fopenmp
CC=gcc

all: KNNserial KNNparallel run

KNNserial: KNNserial.c
	$(CC) KNNserial.c -lm -o KNNserial $(OMPFLAG)

KNNparallel: KNNparallel
	$(CC) KNNparallel.c -lm -o KNNparallel $(OMPFLAG)

run:
	time ./KNNserial 
	time ./KNNparallel

clean:
	rm KNNserial KNNparallel
