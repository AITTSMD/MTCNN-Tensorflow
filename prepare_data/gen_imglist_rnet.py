import numpy as np
import numpy.random as npr
import os

data_dir = '../../DATA'
dir_name = 'no_LM24'

#anno_file = os.path.join(data_dir, "anno.txt")

size = 24

if size == 12:
    net = "PNet"
elif size == 24:
    net = "RNet"
elif size == 48:
    net = "ONet"


with open(os.path.join(data_dir, '%s/pos_%s.txt' % (dir_name, size)), 'r') as f:
    pos = f.readlines()

with open(os.path.join(data_dir, '%s/neg_%s.txt' % (dir_name, size)), 'r') as f:
    neg = f.readlines()

with open(os.path.join(data_dir, '%s/part_%s.txt' % (dir_name, size)), 'r') as f:
    part = f.readlines()


#with open(os.path.join(data_dir, '%s/landmark_%s_aug.txt' % (size, size)), 'r') as f:
#    landmark = f.readlines()
with open(os.path.join(data_dir, '%s/landmark_%s_aug.txt' % (size, size)), 'r') as f:
    landmark = f.readlines()
  
#dir_path = os.path.join(data_dir, 'imglists',"RNet")
dir_path = os.path.join(data_dir, 'imglists_noLM',"RNet")
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
#write all data
with open(os.path.join(dir_path, "train_%s_landmark.txt" % (net)), "w") as f:
    print(len(neg))
    print(len(pos))
    print(len(part))
    print(len(landmark))
    for i in np.arange(len(pos)):
        f.write(pos[i])
    for i in np.arange(len(neg)):
        f.write(neg[i])
    for i in np.arange(len(part)):
        f.write(part[i])
    for i in np.arange(len(landmark)):
        f.write(landmark[i])
