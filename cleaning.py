import os, csv

# Create all training directories

breeds = open("breeds.txt").read().splitlines()
for breed in breeds:
	directory = "./data/train/" + breed
	if not os.path.exists(directory):
		os.makedirs(directory)

# Move files to respective directories

filename = "labels.csv"

with open(filename, 'r') as csv_file:
	csv_reader = csv.DictReader(csv_file)
	line_count = 0
	for row in csv_reader:
		if line_count == 0:
			print(f'Column names are {", ".join(row)}')
			line_count += 1
		print(f'\timage {row["id"]} is a {row["breed"]}')

		# move
		os.rename("./train/" + row["id"] + ".jpg", 
				  "./data/train/" + row["breed"] + "/" + row["id"] + ".jpg")		

		line_count += 1
	print(f'Processed {line_count} lines.')