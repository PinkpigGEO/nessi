#	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/home/pageotd/miniconda3/envs/nessi3.6/include/python3.6m/ -o dsp_masw.so dsp_masw.c

CC = gcc

all:
	cythonize -a -i dsp_masw.pyx
	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/home/pageotd/miniconda3/envs/nessi3.6/include/python3.6m/ -o dsp_masw.so dsp_masw.c
