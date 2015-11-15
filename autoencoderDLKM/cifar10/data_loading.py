import cPickle
import Image
import matplotlib.pyplot as plt

fo = open('data_batch_1', 'rb')
dict = cPickle.load(fo)
fo.close()
print dict.keys()

data = dict['data']
labels = dict['labels']

label = np.asarray(label) #convert the 'list' of labels to ndarray





rgbArray = np.zeros((32,32,3), 'uint8')
rgbArray[..., 0] = data[1][0:1024].reshape(32,32)
rgbArray[..., 1] = data[1024:2048].reshape(32,32)
rgbArray[..., 2] = data[2048:].reshape(32,32)
img = Image.fromarray(rgbArray) #image is generated by fusing all 3 channels
img.save('file.png')
plt.imshow(rgbArray)
plt.show()