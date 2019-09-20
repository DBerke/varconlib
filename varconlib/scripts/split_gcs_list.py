# -*- coding: utf-8 -*-

infile = '/Users/dberke/Documents/GCS_list_HD_numbers.txt'
outfile = '/Users/dberke/Documents/InputLists/GCS_list_HD_strings.txt'

j = 0

with open(infile, 'r') as f:
    lines = f.readlines()
    print(len(lines))
    with open(outfile, 'w') as g:
        for line in lines:
            newline = "HD" + line.strip() + '\n'
            g.write(newline)
            j += 1
            if j % 100 == 0:
                print('Processed {} lines.'.format(j))


