import argparse
import logging
import sys

def main(POP, CXPB, MUTPB, DATFILE):
	sys.stdout.write("YOU ARE IN THE MAIN")
	# just a test
	score = MUTPB*POP/100
	score = float(score)
	score = score - float(CXPB)
	if score < 0:
		score = 0
	
	# save the fo values in DATFILE
	with open(DATFILE, 'w') as f:
		f.write(str(score*100))

if __name__ == "__main__":
	# just check if args are ok
	with open('args.txt', 'w') as f:
		exe = str(sys.argv[0])
		instance = str(sys.argv[3]).split("/")[-1]
		rest_params = sys.argv[4:7]

		to_write = [exe, instance]
		for param in rest_params:
			to_write.append(param)

		last_param = str(sys.argv[7] + "="+ sys.argv[-1])
		to_write.append(last_param)

		f.write(str(to_write))

	# loading example arguments
	"""ap = argparse.ArgumentParser(description='Feature Selection using GA with DecisionTreeClassifier')
	sys.stdout.write("\n	VERBOSE")
	ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
	# 3 args to test values
	sys.stdout.write("\n	POP:" + str(to_write[2]))
	ap.add_argument('--pop', dest='pop', type=int, required=True, help='Population size')
	sys.stdout.write("\n	CROSS:"  + str(to_write[3]))
	ap.add_argument('--cros', dest='cros', type=float, required=True, help='Crossover probability')
	sys.stdout.write("\n	MUT:" + str(to_write[4]))
	ap.add_argument('--mut', dest='mut', type=float, required=True, help='Mutation probability')
	# 1 arg file name to save and load fo value
	sys.stdout.write("\n	DATFILE:" + str(to_write[5]))
	ap.add_argument('--datfile', dest='datfile', type=str, required=True, help='File where it will be save the score (result)')
	"""
	#sys.stdout.write("\n LLEGASTE 3: PARSE ARGS")
	#args = ap.parse_args()
	#sys.stdout.write("\n LLEGASTE 4: DEBUG")
	#logging.debug(args)
	# call main function passing args

	pop = int(str(to_write[2]).split("=")[1])
	cros = float(str(to_write[3]).split("=")[1])
	mut = float(str(to_write[4]).split("=")[1])
	dat = str(to_write[5]).split("=")[1]

	main(pop, cros, mut, dat)