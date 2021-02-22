import base64
import struct
import os
import numpy as np
from PIL import Image
import glob
import pickle

 
def read_line(line):
    m_id, image_search_rank, image_url, page_url, face_id, face_rectangle, face_data=line.split("\t")
    rect=struct.unpack("ffff",base64.b64decode(face_rectangle))
    return m_id, image_search_rank, image_url, page_url, face_id, rect, base64.b64decode(face_data)
 
def write_image(filename, data):
    with open(filename,"wb") as f:
        f.write(data)
 
def unpack(file_name, output_dir):
    i=0
    j=0
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            m_id, image_search_rank, image_url, page_url, face_id, face_rectangle, face_data = read_line(line)
            img_dir = os.path.join(output_dir, m_id)
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            img_name = "%s-%s" % (image_search_rank, face_id) + ".jpg"
            img_path = os.path.join(img_dir, img_name)
            write_image(img_path, face_data)
            
            img = Image.open(img_path)
            fail = 0
            if (img.format != 'JPEG'):
                fail = 1
            try:
                img.verify()
            except Exception:
                fail = 1
            if fail == 1:
                print('%s is not a jpeg!' % img_name)
                os.remove(img_path)
                j += 1
            
            i += 1
            if i % 1000 == 0:
                print(i, "images finished", j, "images failed")
        print("all finished")
 
def main():
    file_name = "MS-Celeb-1M/data/aligned_face_images/FaceImageCroppedWithAlignment.tsv"
    output_dir = "MS-Celeb-1M/FaceImageCroppedWithAlignment"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #unpack(file_name, output_dir)
    image_files = []
    for dir, _, _, in os.walk(output_dir):
        filenames = glob.glob( os.path.join(dir, '*.jpg'))  # may be JPEG, depending on your image files
        image_files.append(filenames)
    np.random.seed(123)
    rp = np.random.permutation(len(image_files)).tolist()
    image_files_train = [image_files[j] for j in rp[:73678]]
    image_files_val = [image_files[j] for j in rp[73678:83688]]
    image_files_test = [image_files[j] for j in rp[83688:]]
    image_files_train = np.hstack(image_files_train)
    image_files_val = np.hstack(image_files_val)
    image_files_test = np.hstack(image_files_test)
    pickle.dump( {'image_path' : image_files_train}, open( os.path.join(output_dir, 'train_filename.pickle'), "wb" ) )
    pickle.dump( {'image_path' : image_files_val}, open( os.path.join(output_dir, 'valid_filename.pickle'), "wb" ) )
    pickle.dump( {'image_path' : image_files_test}, open( os.path.join(output_dir, 'test_filename.pickle'), "wb" ) )

if __name__ == '__main__':
    main()

