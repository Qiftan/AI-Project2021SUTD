import io
import re
import pandas as pd



def make_data(fromFilename, start=0):  
    MAXLINES = 10000
    csvfile = open(fromFilename, mode='r', encoding='utf-8')
    # or 'Latin-1' or 'CP-1252'
    filename = start
    for rownum, line in enumerate(csvfile):
        if rownum % MAXLINES == 0:
            filename += 1
            outfile = open(datapath +"/covid_data_" + str(filename) + '.csv', mode='w', encoding='utf-8')
        outfile.write(re.sub('\t',',',re.sub('(^|[\t])([^\t]*\,[^\t\n]*)', r'\1"\2"', line)))
        # outfile.write(line)
    outfile.close()
    csvfile.close()

datapath = 'data2_2'

if __name__ == "__main__":
    # make_data('raw_data/TweetsCOV19.tsv')
    make_data('raw_data/TweetsCOV19_2.tsv')
    # make_data('raw_data/TweetsCOV19_3.tsv')