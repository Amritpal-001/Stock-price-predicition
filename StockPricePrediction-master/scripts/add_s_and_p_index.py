import os, sys, csv
import math
import pprint

def roundup(var):
	return float(format(var, '.6f'))

def main(dir_path, sp_index_file, output_dir):
	files = os.listdir(dir_path)

	for file_name in files:
		with open( os.path.join(dir_path, file_name), 'r') as textfile:

			new_file = open(os.path.join(output_dir, file_name), 'w+')

			new_list = []
			new_list.append(['symbol','date','open','high','low','close','volume','adj_close', 'prev_day_diff', '50_day_moving_avg', '10_day_volatility', 
				's&p_index_open', 's&p_index_high', 's&p_index_low', 's&p_index_close', 's&p_index_volume', 's&p_index_adj_close'])

			dict_mapping = {}

			for count, row in enumerate(reversed(list(csv.reader(textfile)))):
				if str(row[0])=="symbol":
					break

				date = str(row[1])
				dict_mapping[date] = row

			"""
				Extend to Existing Key-Value in dict_mapping dictionary.
			"""

			with open(sp_index_file, 'r') as sp_index_fp:
				for count2, row2 in enumerate(reversed(list(csv.reader(sp_index_fp)))):
					if str(row2[0]) in dict_mapping:
						dict_mapping[str(row2[0])].extend(row2[1:])

			#pprint.pprint(dict_mapping, width=1)

			for key in sorted(dict_mapping):
				new_list.append(dict_mapping[key])
			
			writer = csv.writer(new_file)
			writer.writerows(new_list)
			new_file.close()
		textfile.close()

if __name__ == '__main__':
	main(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]))