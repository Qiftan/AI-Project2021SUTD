def split_data(fromFilename):  
    MAXLINES = 100000
    csvfile = open(fromFilename, mode='r', encoding='utf-8')
    # or 'Latin-1' or 'CP-1252'
    filename = 0
    for rownum, line in enumerate(csvfile):
        if rownum % MAXLINES == 0:
            filename += 1
            outfile = open("covid_data_" + str(filename) + '.tsv', mode='w', encoding='utf-8')
        outfile.write(line)
    outfile.close()
    csvfile.close()

if __name__ == "__main__":
    split_data('./TweetsCOV19.tsv')