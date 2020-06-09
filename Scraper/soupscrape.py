#!/usr/local/bin/python3

# Imports
import smtplib
import urllib.request
from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen
from nltk.tokenize import regexp_tokenize
import lxml.html

headers = {"User-Agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36'}


url = "https://www.bailii.org/ew/cases/EWCA/Crim/2008/"

from urllib.request import Request, urlopen

req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
response = urlopen(req, timeout=20).read()
soup = BeautifulSoup(response, features='html.parser')
all_links = soup.find_all('a')

for link in all_links[13:32]:
    check_url = link['href']
    finalurl = "https://www.bailii.org" + check_url
    page = urllib.request.Request(finalurl, headers={'User-Agent': 'Mozilla/5.0'})
    html = urlopen(page)
    bs = BeautifulSoup(html, "html.parser")
    body = bs.find_all('ol')
    body_str = str(body)

    soup = BeautifulSoup(body_str, 'lxml')
    text = soup.get_text()
    f = open("2008crim2.txt", "a")
    print(text, file=f)
    f.close()


