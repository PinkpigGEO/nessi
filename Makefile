all:
	cd nessi/modbuilder/ && cythonize -a -i *.pyx && cd -
	cd nessi/modeling/swm/ && cythonize -a -i *.pyx && cd -
