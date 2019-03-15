all:
	cd nessi && make && cd -

clean:
	cd nessi/modbuilder/ && cythonize -a -i *.pyx && cd -
	cd nessi/modeling/swm/ && cythonize -a -i *.pyx && cd -
