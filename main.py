import argparse
import logging
import sys

def main(POP, CXPB, MUTPB, DATFILE):
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

	pop = int(str(to_write[2]).split("=")[1])
	cros = float(str(to_write[3]).split("=")[1])
	mut = float(str(to_write[4]).split("=")[1])
	dat = str(to_write[5]).split("=")[1]

	main(pop, cros, mut, dat)