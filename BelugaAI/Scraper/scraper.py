import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import regexp_tokenize
import os,sys
import subprocess
import glob
from os import path
import re

mystring = open('1976.txt', 'r')
mystring = mystring.read().splitlines()
mystr = str(mystring)


pattern = "\s+[^.!?]*\[\d+\].*?[.!?]|\s+[^.!?]*\(\d+\).*?[.!?]"

case1 = regexp_tokenize(mystr, pattern)

case = str(case1)
case_newline = case.replace("', '", " ")
case_newline2 = case_newline.replace(". ", "\n")
case3 = case_newline2.replace("[", " ")
case4 = case3.replace("\\", "")
case5 = case4.replace("',", "")
case6 = case5.replace(" '", "")
case7 = case6.replace('",', "")




file = open("scraped3.txt", "w")
file.write(case7)
file.close()



