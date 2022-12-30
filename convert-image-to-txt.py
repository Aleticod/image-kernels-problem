import sys
import matplotlib.image as image

# Get the name of the image
file_name = sys.argv[1]
file_dir = './images/' + file_name

# Read the image
img = image.imread(file_dir)
height, width = img.shape
print("Image shape width x height: ", width, height)

# Direction to save the result
result_dir = './results/txt_' + file_name[:-3] + "txt" 
data_dir = './results/data_' + file_name[:-3] + "txt"

# Write a new file with pixels from image
with open(result_dir, 'w') as f:
    for line in img:
        for pix in line:
            f.write(str(pix) + " ")
        f.write('\n')

# Write a new file with data from image
with open(data_dir, 'w') as data:
    data.write(file_name[:-4] + " ")    #name of the image
    data.write(file_name[-3:] + " ")    #extension of the image
    data.write(str(width) + " ")        #width of the image
    data.write(str(height) + " ")       #height of the image 
    data.write(sys.argv[2])             #kernel number


