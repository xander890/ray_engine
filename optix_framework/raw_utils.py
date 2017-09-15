import numpy as np
import struct
import sys

def read_raw(file):
    """ Reads a .raw/.txt file couple as a hxwx3 numpy array."""
    if file[-4:] == '.raw':
       data_file = file
       txt_file = file[:-4] + '.txt'
    elif file[-4:] == '.txt':
       data_file = file[:-4] + '.raw' 
       txt_file = file
    else:
       data_file = file[:-4] + '.raw' 
       txt_file = file[:-4] + '.txt'
       
    with open(data_file, 'rb') as f:
        data = f.read()

    with open(txt_file, 'r') as f:
        text = f.read()

    # Parsing txt file.
    sizes = text.split('\n')[1].split(" ")
    w,h = [int(s) for s in sizes]

    # Parsing data file.
    myfmt='f'*(w*h*3)
    unpacked = np.array(struct.unpack(myfmt,data)) 

    return np.flipud(np.reshape(unpacked, [h,w,3]))
    
def save_raw(filename, data):
    if(len(data.shape) != 3) or len(filename) < 4:
        print("Error, only RGB images are accepted")
        return 
    data = np.flipud(data)
    if filename[-4:] == '.raw' or filename[-4:] == '.txt':
        filename = filename[:-4]

    raw_filename = filename + '.raw'
    txt_filename = filename + '.txt'
    w = data.shape[1]
    h = data.shape[0]
    txt_content = "0\n" + str(w) + " " + str(h)
    with open(txt_filename, 'w') as f:
        f.write(txt_content)

    imgRawData = data.flatten()
    myfmt='f'*len(imgRawData)
    bin=struct.pack(myfmt,*imgRawData)
     
    with open(raw_filename, 'wb') as f:
        f.write(bin)
        
        
if __name__ == "__main__":
    from PIL import Image     
    data = read_raw('test.raw')
    image = np.clip(data, 0.0, 1.0) * 255.0
    image_tm = image.astype('uint8')
    img = Image.fromarray(image_tm, 'RGB')
    img.save('test.png')
    save_raw('test2.raw', data)