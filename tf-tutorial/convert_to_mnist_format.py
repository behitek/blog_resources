#
# This python script converts a sample of the notMNIST dataset into 
# the same file format used by the MNIST dataset. If you have a program
# that uses the MNIST files, you can run this script over notMNIST to
# produce a new set of notMnist files that should be compatible with
# your program.
#
# Instructions:
#
# 1) if you already have a MNIST notMnist/ directory, rename it and create
#    a new one
#
# $ mv notMnist notMnist.original_mnist
# $ mkdir notMnist
#
# 2) Download and unpack the notMNIST notMnist. This can take a long time
#    because the notMNIST notMnist set consists of ~500,000 files
#
# $ curl -o notMNIST_small.tar.gz http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz
# $ curl -o notMNIST_large.tar.gz http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz
# $ tar xzf notMNIST_small.tar.gz
# $ tar xzf notMNIST_large.tar.gz
#
# 3) Run this script to convert the notMnist to MNIST files, then compress them.
#    These commands will produce files of the same size as MNIST
#    notMNIST is larger than MNIST, and you can increase the sizes if you want.
#
# $ python convert_to_mnist_format.py notMNIST_small 1000 notMnist/t10k-labels-idx1-ubyte notMnist/t10k-images-idx3-ubyte
# $ python convert_to_mnist_format.py notMNIST_large 6000 notMnist/train-labels-idx1-ubyte notMnist/train-images-idx3-ubyte
# $ gzip notMnist/*ubyte
# 

import numpy, imageio, glob, sys, os, random

def get_labels_and_files(folder, number):
  # Make a list of lists of files for each label
  filelists = []
  for label in range(0,10):
    filelist = []
    filelists.append(filelist);
    dirname = os.path.join(folder, chr(ord('A') + label))
    for file in os.listdir(dirname):
      if (file.endswith('.png')):
        fullname = os.path.join(dirname, file)
        if (os.path.getsize(fullname) > 0):
          filelist.append(fullname)
        else:
          print('file ' + fullname + ' is empty')
    # sort each list of files so they start off in the same order
    # regardless of how the order the OS returns them in
    filelist.sort()

  # Take the specified number of items for each label and
  # build them into an array of (label, filename) pairs
  # Since we seeded the RNG, we should get the same sample each run
  labelsAndFiles = []
  for label in range(0,10):
    filelist = random.sample(filelists[label], number)
    for filename in filelist:
      labelsAndFiles.append((label, filename))

  return labelsAndFiles

def make_arrays(labelsAndFiles):
  images = []
  labels = []
  for i in range(0, len(labelsAndFiles)):

    # display progress, since this can take a while
    if (i % 100 == 0):
      sys.stdout.write("\r%d%% complete" % ((i * 100)/len(labelsAndFiles)))
      sys.stdout.flush()

    filename = labelsAndFiles[i][1]
    try:
      image = imageio.imread(filename)
      images.append(image)
      labels.append(labelsAndFiles[i][0])
    except:
      # If this happens we won't have the requested number
      print("\nCan't read image file " + filename)

  count = len(images)
  imagedata = numpy.zeros((count,28,28), dtype=numpy.uint8)
  labeldata = numpy.zeros(count, dtype=numpy.uint8)
  for i in range(0, len(labelsAndFiles)):
    imagedata[i] = images[i]
    labeldata[i] = labels[i]
  print("\n")
  return imagedata, labeldata

def write_labeldata(labeldata, outputfile):
  header = numpy.array([0x0801, len(labeldata)], dtype='>i4')
  with open(outputfile, "wb") as f:
    f.write(header.tobytes())
    f.write(labeldata.tobytes())

def write_imagedata(imagedata, outputfile):
  header = numpy.array([0x0803, len(imagedata), 28, 28], dtype='>i4')
  with open(outputfile, "wb") as f:
    f.write(header.tobytes())
    f.write(imagedata.tobytes())
    


def main(argv):
  # Uncomment the line below if you want to seed the random
  # number generator in the same way I did to produce the
  # specific notMnist files in this repo.
  # random.seed(int("notMNIST", 36))

  labelsAndFiles = get_labels_and_files(argv[1], int(argv[2]))
  random.shuffle(labelsAndFiles)
  imagedata, labeldata = make_arrays(labelsAndFiles)
  write_labeldata(labeldata, argv[3])
  write_imagedata(imagedata, argv[4])

if __name__=='__main__':
  main(sys.argv)
