#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:29:49 2020

Scraper for dl all html-formatted fillings on https://www.sec.gov/Archives/edgar/data/51143/ and saving them as pdfs

Format of output file: nameoffolder_nameoffile_typeoffilling.pdf

@author: odrec
"""

import requests
import re, pdfkit
from bs4 import BeautifulSoup

root_url = 'https://www.sec.gov/'
url = 'https://www.sec.gov/Archives/edgar/data/51143/'
response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")
alllinks = soup.findAll('a')
alllinks = alllinks[54:6781]
#to check if all files were downloaded
files_names = []
for l in alllinks:
    one_a_tag = l
    link = one_a_tag['href']
    file = link.split('/')[-1]
    files_names.append(file)
    print("\nProcessing file", len(files_names))
    print("Name of file", file)
    
    url_down = url + file + "/"
    response_down = requests.get(url_down)
    soup_down = BeautifulSoup(response_down.text, "html.parser")
    alllinks_down = soup_down.findAll('a')
    index_links = list(filter(lambda x: 'index' in str(x), alllinks_down))
    edgar_links = list(filter(lambda x: 'edgar' in str(x), index_links))
    link = str(list(filter(lambda x: not 'headers' in str(x), edgar_links))[0])
    link = link.split('"')[1]
    link = link.split('/')[-1]
    
    url_down_down = url_down + link
    response_down_down = requests.get(url_down_down)
    soup_down_down = BeautifulSoup(response_down_down.text, "html.parser")
    
    #Get the html links for the files
    alllinks_down_down = soup_down_down.findAll('a')
    edgar_links = list(filter(lambda x: 'edgar' in str(x), alllinks_down_down))
    data_links = list(filter(lambda x: 'data' in str(x), edgar_links))
    html_links = list(filter(lambda x: 'htm' in str(x), data_links))
    
    #Get the type of filling
    alldivs = soup_down_down.findAll('div')    
    div_info = list(filter(lambda x: 'companyInfo' in str(x), alldivs))
    company_info = list(filter(lambda x: 'Type:' in str(x), div_info))
    type_info = re.split('<strong>|</strong>', str(company_info[0])) 
    
    indices = [i for i, s in enumerate(type_info) if 'Type:' in s]
        
    #sometimes the type has unnecessary spaces
    typ = type_info[indices[0]+1].strip()
    #some types have / separators that get mistaken by paths in the url so I substitute them by a -
    typ = typ.replace('/', '-')
    for hl in html_links:
        hl = str(hl)
        htm_name = re.split('<|>', hl) 
        if '.htm' in htm_name[2] or '.html' in htm_name[2]:
            html_url = htm_name[1].split('"')
            file_name = html_url[1].split('/')[-1]
            
            #This is done to avoid using Inline XBRL Viewer when the file is set to open with it automatically
            html_path = html_url[1].split('=')
            if len(html_path) > 1: html_path = html_path[1]
            else: html_path = html_path[0]
            
            html_url = root_url + html_path
    
            file_name_pdf = file+'_'+file_name.split('.')[0]+'_'+typ+'.pdf'
            pdfkit.from_url(html_url, 'files/'+file_name_pdf)
            print("Saved file from file", len(files_names))

