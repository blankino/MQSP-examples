#!/usr/bin/env wolframscript
(* ::Package:: *)

\[Gamma]0=\[Alpha]0*k*I;
\[Gamma]1=\[Alpha]1/(k*I);
AA = 1;
FF = 1;
BB = -(\[Alpha]0+\[Alpha]1);
CC = \[Alpha]0*\[Alpha]1;
DD=-(\[Gamma]0+\[Gamma]1);
GG=-(\[Alpha]0+Conjugate[\[Alpha]1]);
HH=\[Alpha]0*Conjugate[\[Alpha]1];
II=-(\[Gamma]0+Conjugate[\[Gamma]1]);
r=Conjugate[GG]*II+Conjugate[II]*GG-BB*Conjugate[DD]-DD*Conjugate[BB];
s=-(Conjugate[GG]*II+GG*Conjugate[II]+BB*Conjugate[DD]+DD*Conjugate[BB]);
t=-(Conjugate[AA]*BB+Conjugate[BB]*AA+BB*Conjugate[CC]+Conjugate[BB]*CC+FF*Conjugate[GG]+GG*Conjugate[FF]+GG*Conjugate[HH]+Conjugate[GG]*HH);
u=II*Conjugate[HH]+HH*Conjugate[II]-FF*Conjugate[II]-Conjugate[FF]*II-AA*Conjugate[DD]-DD*Conjugate[AA]-CC*Conjugate[DD]-Conjugate[CC]*DD;
x=(DD*Conjugate[AA]-AA*Conjugate[DD])*(r*CC-s*AA)-(r*DD-t*AA)*(Conjugate[AA]*CC-Conjugate[CC]*AA);
y=(r*DD-t*AA)*(BB-Conjugate[BB])-(r*BB-u*AA)*(DD-Conjugate[DD]);
k=3;
\[Alpha]0=1+I;
Clear[\[Alpha]1];
Solve[x==0&&y==0,\[Alpha]1,Complexes]



