
%{
	#include "stdio.h"
	extern FILE * yyin;
	extern char* yytext;
	extern int yyparse();
	extern int yylex();
	extern int yylineno;
	char * s_node;
	char * t_node;
	char ** buffer[1000];
	int dec_ind[1000];
	int fun_count[1000];
	int count = 0;
	int d_count = 0;
	int f_count = 0;
	int f_ind = 0;
	char * str_con(char *s1,char * s2);
	//dot -Tpng out.dot > graph.png
%}


%union{
	char *sval;
	int ival;
}

//tokens
%token INT_T CHAR_T FLOAT_T BOOL_T LONG_T DOUBLE_T VOID_T
%token INT_V CHAR_V FLOAT_V BOOL_V 
%token <sval> ID 
%token ASSIGN
%token ADD SUB MUL DIV
%token FOR WHILE IF ELSE ELSEIF
%token BREAK CONTINUE RETURN
%token ANDP ORP NOTP
%token AND OR NOT XOR
%token EQ NE LE GE LT GT
%token STRUCT
%token TRUE FALSE 
%token SC LB RB LP RP LC RC CM
%token INC DEC


%start Program

//Grammar defined here
%%

Program : F_dec | L_stmt | S_dec |;
Stmt : Access | Dec | While_s | For_s | If_s | F_call SC | cExp SC | sExp SC;
C_stmt : LC L_stmt RC ;
L_stmt : Stmt L_stmt |;

S_dec: STRUCT ID LC L_stmt RC SC Program;

fC_stmt : LC L_stmt RC {fun_count[f_ind]=f_count;f_count=0;f_ind++;};

Type: INT_T | CHAR_T | FLOAT_T | VOID_T | BOOL_T | LONG_T | DOUBLE_T;
Opt : ADD | SUB | MUL | DIV | AND | OR | NOT | XOR | ANDP | ORP 
	| NOTP | EQ | NE | LE | GE | LT | GT;
Exp : ID | Exp Opt Exp | F_call | ID LB INT_V RB | MUL ID | AND ID | Value | LP Exp RP | NOTP Exp;
cExp : ID INC | ID DEC | INC ID | DEC ID;
sExp : RETURN | CONTINUE | BREAK | RETURN Exp;
Value : INT_V | CHAR_V | FLOAT_V | BOOL_V;

F_dec : Type ID LP Fd_args RP fC_stmt Program {buffer[count]= $2;count++; dec_ind[d_count] = count; d_count++;};
F_call : ID LP Fc_args RP {buffer[count] = $1; count++;f_count++;};
Fc_args : Exp CM Fc_args | Exp |;
Fd_args : Type ID CM Fd_args| Type ID |;

Arr_val : Value CM Arr_val | Value;

Dec: Type ID SC | Type ID ASSIGN Exp SC | Type ID LB INT_V RB SC | Type MUL ID SC | Type MUL Access
	| Type ID LB INT_V RB ASSIGN LC Arr_val RC SC | F_dec | S_dec;

Access : ID ASSIGN Exp SC | ID LB INT_V RB ASSIGN Exp SC;
f_Access : ID ASSIGN Exp | cExp;

While_s: WHILE LP Exp RP Stmt | WHILE LP Exp RP C_stmt;

For_s: FOR LP Type ID ASSIGN Exp SC Exp SC f_Access RP C_stmt | FOR LP Type ID ASSIGN Exp SC Exp SC f_Access RP
	| FOR LP ID ASSIGN Exp SC Exp SC f_Access RP C_stmt | FOR LP ID ASSIGN Exp SC Exp SC f_Access RP; 

If_s: IF LP Exp RP Stmt elif | IF LP Exp RP C_stmt elif
	| IF LP Exp RP Stmt | IF LP Exp RP C_stmt;

elif : ELSE Stmt | ELSE C_stmt | ELSEIF LP Exp RP Stmt elif 
	| ELSEIF LP Exp RP C_stmt elif;

%%

#include<ctype.h>

main(void){
	FILE * fptr = fopen("test.c","r");
	FILE * fptr2 = fopen("out.dot","w");
	fprintf(fptr2,"digraph x{\n");
	yyin = fptr;
	do{
		yyparse();
	}while(!feof(yyin));
	printf("end....\n");
	fclose(yyin);
	int sep_point = dec_ind[0]-1;
	int first_dec = count-1;
	int vc = 0;
	int kc = 0;
	for(int i=0;i<sep_point;i++){
		if(vc == fun_count[kc]){
			first_dec--;
			vc=0;
			kc++;
		}
		fprintf(fptr2,"%s->%s;\n",buffer[first_dec],buffer[i]);
		vc++;
	}
	fprintf(fptr2,"}");
	fclose(fptr2);
}

yyerror(char *s){
  printf("error...\n");
}
