import os, sys, csv
import math

def roundup(var):
	return float(format(var, '.6f'))

def main(dir_path, output_dir):
	files = os.listdir(dir_path)
	for file_name in files:
		with open( os.path.join(dir_path, file_name), 'r') as textfile:
			new_file = open(os.path.join(output_dir, file_name), 'w+')
			new_list = []

			prev = 0.0
			diff = 0.0
			avg = 0.0
			num_moving_avg = 50
			volatile_avg = 0.0
			num_volatile = 10
			curr_volatility = 0.0

			for count, row in enumerate(reversed(list(csv.reader(textfile)))):
				if not count:
					try:
						row[8]=prev
					except Exception, e:
						row.append(prev)
				else:
					diff = roundup(float(row[7]) - float(prev))
					try:
						row[8]=diff
					except Exception, e:
						row.append(diff)
				
				if count<num_moving_avg:
					avg = roundup((count * avg + float(row[7]))/ (count + 1))
				else:
					avg = roundup((num_moving_avg * avg + float(row[7]) - float(new_list[count - num_moving_avg][7])) / (num_moving_avg)) 

				prev = float(row[7])
				
				if count < num_volatile:
					volatile_avg = roundup((count * volatile_avg + float(row[7]))/ (count + 1))
				else:
					volatile_avg = roundup((num_volatile * volatile_avg + float(row[7]) - float(new_list[count - num_volatile][7])) / (num_volatile))

				if count:
					loop_count = min(count, num_volatile)
				
					for i in range(loop_count):
						curr_volatility += math.pow((float(row[7]) - volatile_avg), 2)

					curr_volatility = roundup(math.sqrt(curr_volatility / (loop_count))) 
				
				try:
					row[9]=avg
					row[10]=curr_volatility
				except Exception, e:
					row.append(avg)
					row.append(curr_volatility)

				new_list.append(row)
				curr_volatility = 0.0

			new_list.insert(0, ['symbol','date','open','high','low','close','volume','adj_close', 'prev_day_diff', '50_day_moving_avg', '10_day_volatility'])

			writer = csv.writer(new_file)
			writer.writerows(new_list)
			new_file.close()
		textfile.close()

if __name__ == '__main__':
	main(str(sys.argv[1]), str(sys.argv[2]))