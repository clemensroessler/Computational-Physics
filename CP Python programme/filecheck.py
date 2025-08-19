#!/usr/bin/env python3
"""Report or replace tabs in python files and check for
   Check for lines longer than 80 characters

Usage::

    python3 filecheck.py -f yourfile.py [-o youroutput.py]
"""

import os
import sys
from optparse import OptionParser


def report(datastring):                                 # pylint: disable=R0912
    """ Report only on long lines, non-ascii's, and tabs.
    """
    lines = datastring.splitlines()
    tabs, nonasciis, lengths = [], [], []

    # check every line and save for report
    for i in range(len(lines)):
        if "\t" in lines[i]:
            tabs.append(i + 1)

        # Check if lines are not too long:
        if len(lines[i]) >= 80:
            lengths.append(i + 1)

    # report results:
    if len(tabs) > 0:
        print("TAB's found in lines:")
        for line in tabs:
            print("line ", line)

    if len(lengths) > 0:
        print("lines too long:")
        for line in lengths:
            print("line ", line)

    if len(tabs) + len(nonasciis) + len(lengths) == 0:
        print("Nothing found.")


def replace(datastring):
    """ Replace non-ascii's and tabs. Report on long lines.
    """

    datastring = datastring.replace("\t", "    ")  # replace tab's

    # check line length and remove trailing whitespaces
    lines = datastring.splitlines()
    lengths = []

    for i in range(len(lines)):
        lines[i] = lines[i].rstrip()
        if len(lines[i]) > 80:
            lengths.append(i + 1)

    datastring = ""
    for line in lines:
        datastring += line + "\n"

    if len(lengths) > 0:
        print("lines too long (>= 80 characters):")
        for line in lengths:
            print("line ", line)

    return datastring


def main():
    """ Main routine: reads parameters from the command line
        or by using wx-dialogs
    """
    # parse any command line options
    parser = OptionParser()
    parser.add_option("-f", "--file", action="store",
                      type="string", default=None,
                      dest="inputfile", metavar="filename.py",
                      help="file to be checked")

    parser.add_option("-o", "--output", action="store",
                      type="string", default=None,
                      dest="outputfile", metavar="filename_replaced.py",
                      help="file to save changes (optional)")

    options, _args = parser.parse_args()

    if options.inputfile is None:                # input filename is required
        parser.print_help()
        sys.exit(1)

    fsrc = open(options.inputfile, 'r')          # open source for reading
    contents = fsrc.read()                       # read the whole source file
    fsrc.close()                                 # close file

    if options.outputfile is None:
        report(contents)                         # only report

    else:
        report(contents)                         # report and
        contents = replace(contents)             # change data

        if os.path.exists(options.outputfile):
            try:
                os.remove(options.outputfile + "~")
            except OSError:
                pass
            os.rename(options.outputfile, options.outputfile + "~")

        fdst = open(options.outputfile, 'w')     # open destination for writing
        fdst.write(contents)                     # write new file
        fdst.close()                             # close file

        print()
        print("Wrote: {}".format(options.outputfile))

    input("Press <ENTER>!")                      # keep terminal open (windows)


if __name__ == "__main__":
    main()