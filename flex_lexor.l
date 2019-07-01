%{
	#include "sk.tab.h"
	#include "stdio.h"
	//for debug printf("117\n");
%}

%%
";" {return SC;}
"[" {return LB;}
"]" {return RB;}
"(" {return LP;}
")" {return RP;}
"{" {return LC;}
"}" {return RC;}
"," {return CM;}
"++" {return INC;}
"--" {return DEC;}
"int" {return INT_T;}
"char" {return CHAR_T;}
"float" {return FLOAT_T;}
"bool" {return BOOL_T;}
"long" {return INT_T;}
"double" {return FLOAT_T;}
"void" {return VOID_T;}
"for" {return FOR;}
"while" {return WHILE;}
"if" {return IF;}
"else if" {return ELSEIF;}
"else" {return ELSE;}
"break" {return BREAK;}
"continue" {return CONTINUE;}
"return" {return RETURN;}
"struct" {return STRUCT;}
"+" {return ADD;}
"-" {return SUB;}
"*" {return MUL;}
"/" {return DIV;}
"&&" {return ANDP;}
"||" {return ORP;}
"!" {return NOTP;}
"&" {return AND;}
"|" {return OR;}
"~" {return NOT;}
"^" {return XOR;}
"==" {return EQ;}
"!=" {return NE;}
"<=" {return LE;}
">=" {return GE;}
"<" {return LT;}
">" {return GT;}
"True" {return TRUE;}
"False" {return FALSE;}
[A-Za-z_][A-Za-z0-9_]* {yylval.sval=strdup(yytext); return ID;} 
[0-9]+ {yylval.ival=atoi(yytext); return INT_V;}
[A-Za-z] {return CHAR_V;}
[0-9]+"."[0-9]+ {return FLOAT_V;} 
"=" {return ASSIGN;}
[ \t\n] ;
[ \t] ;
\/\/.* ;
. ;
%%
