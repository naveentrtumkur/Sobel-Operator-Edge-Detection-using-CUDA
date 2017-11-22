# Use nvcc for compilation.
CC := nvcc

#flags ro remove warning during compilation.
flags := -arch=sm_60 -O -Wno-deprecated-gpu-targets

#header file to be included.
Head := read_bmp.h 


all: lab4p1 lab4p2

lab4p1:
	$(CC) -O -o lab4p1 tumkurrameshbabu_naveen_lab4p1.cu $(flags)
lab4p2:$(Head)
	$(CC) -O -o lab4p2 tumkurrameshbabu_naveen_lab4p2.cu bmpReader.o $(flags)
clean :
	rm -rf ./lab4p1 ./lab4p2

