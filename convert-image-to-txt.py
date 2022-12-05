import sys
import matplotlib.image as image


file_name = sys.argv[1]
file_dir = './images/' + file_name

#read de image
img = image.imread(file_dir)
size_image = len(img)

# Direction to save the result
result_dir = './results/txt_' + file_name[:-3] + "txt"
data_dir = './results/data_' + file_name[:-3] + "txt"

#write a new file with pixels from image
with open(result_dir, 'w') as f:
    for line in img:
        for pix in line:
            f.write(str(pix) + " ")
        f.write('\n')

with open(data_dir, 'w') as data:
    data.write(file_name[:-4] + " ")
    data.write(file_name[-3:] + " ")
    data.write(str(size_image) + " ")
    data.write(sys.argv[2])


