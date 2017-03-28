from numpy import array


# read data to a list
lines = list(open("/home/ubuntu/Downloads/glove.6B.50d.txt").readlines())
line_list = [line.strip() for line in lines]
line_split = [line.split() for line in line_list]
vocabulary_list= [line[0] for line in line_split]

vocabulary_dict = {x: i for i, x in enumerate(vocabulary_list)}

# write list to file
# with open("/home/ubuntu/Downloads/dict.txt", 'w') as f:
# 	for item in vocabulary_list:
# 		f.write(item + '\n')

# write dict to file
with open("/home/ubuntu/Downloads/dict_file.txt", 'w') as f:
	f.writelines('{} {}\n'.format(k,v) for k, v in vocabulary_dict.items())

# for key in vocabulary_dict:
#     print key, vocabulary_dict[key]
