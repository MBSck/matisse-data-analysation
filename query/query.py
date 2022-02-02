import argparse
import numpy as np
from astroquery.eso import Eso

# Initialize Eso and argparse
parser = argparse.ArgumentParser("Query that can be customized to directly load the wanted observations to the NoMachine virtual environment")
eso = Eso()

# Required argument group
requiredName = parser.add_argument_group("required arguments")

# Parser arguments for log-in
parser.add_argument("--user", "-u", type=str, required=False, default="MbS", help="changes the default user name")

# Parser arguments for query
parser.add_argument("--instrument", "-i", type=str, required=False, default="matisse", help="changes the default instrument")
parser.add_argument("--target", "-tar", type=str, required=False, default="", help="target source for query")
requiredName.add_argument("--stime", "-st", type=str, required=True, help="star time of the observation. Format YYYY-MM-DD")
requiredName.add_argument("--etime", "-et", type=str, required=True, help="end time of observation. Format YYYY-MM-DD")

# Parser argument for help query
parser.add_argument("--help_eso", "-he", required=False, default=False, action="store_true", help="shows the query-options for the instrument")

# Parser arguments for action after query
parser.add_argument("--print", "-p", required=False, default=False, action="store_true", help="prints the observation table")
parser.add_argument("--print_header", "-ph", required=False, default=False, action="store_true", help="prints the headers")
parser.add_argument("--download", "-d", required=False, nargs=2, type=int, help="downloads and decompresses the data set. Takes the data format [<Begin of download>, <End of download>]")

# Adds the parser arguments
args = parser.parse_args()

# Log into Eso
eso.login(args.user)

# Return all the results
eso.ROW_LIMIT = -1

# Query for the parameters
table = eso.query_instrument(args.instrument, column_filters={"target": args.target, "stime": args.stime, "etime": args.etime}, columns=[""], help=args.help_eso)

# Prints the table if print is called
if args.print is True:
	print(table)

if args.print_header is True:
	print(eso.get_headers(table["DP.ID"]))

# data_files = eso.retrieve_data(table["DP.ID"][8:9], destination="/home/scheuck/MATISSE/rawData", continuation=True, unzip=False)
