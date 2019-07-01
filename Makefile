sk:	sk.l sk.y
	bison -d sk.y
	flex sk.l
	gcc -o $@ sk.tab.c lex.yy.c -lfl

c:	c.l c.y
	bison -d c.y
	flex c.l
	gcc -o $@ c.tab.c lex.yy.c -lfl

