import csv 
import matplotlib.pyplot as plt
import numpy as np

f = open('train.csv')
csv = csv.reader(f, delimiter=' ')
lines = [row for row in csv]
for line in lines:
    line[0] = (line[0].split(','))[1]
lines = lines[1:]

labels = ['0.  Nucleoplasm',
          '1.  Nuclear membrane',
          '2.  Nucleoli',
          '3.  Nucleoli fibrillar center',
          '4.  Nuclear speckles',
          '5.  Nuclear bodies',
          '6.  Endoplasmic reticulum',
          '7.  Golgi apparatus',
          '8.  Peroxisomes',
          '9.  Endosomes',
          '10.  Lysosomes ',
          '11.  Intermediate filaments',
          '12.  Actin filaments',
          '13.  Focal adhesion sites',
          '14.  Microtubules',
          '15.  Microtubule ends',
          '16.  Cytokinetic bridge',
          '17.  Mitotic spindle',
          '18.  Microtubule organizing center',
          '19.  Centrosome',
          '20.  Lipid droplets',
          '21.  Plasma membrane',
          '22.  Cell junctions',
          '23.  Mitochondria',
          '24.  Aggresome',
          '25.  Cytosol',
          '26.  Cytoplasmic bodies',
          '27.  Rods & rings']

counter = np.zeros((28))
for line in lines:
    for i in line:
        counter[int(i)] = counter[int(i)]+1


# make pie chart
'''
norm_counter = counter/sum(counter)
others = 0.0
others_num = 0
size = []
new_label = []
for i in range(len(norm_counter)):
    if norm_counter[i] <= 0.03:
        others = others+norm_counter[i]
        norm_counter[i] = 0.0
        others_num = others_num+1
    else:
        size.append(norm_counter[i])
        new_label.append(labels[i])

size.append(others)
new_label.append('Other Proteins')
size = np.array(size)
print(size)

patches,l_text,p_text = plt.pie(size,labels=new_label,labeldistance = 1.1,autopct = '%3.1f%%',shadow = False,startangle = 90,pctdistance = 0.8)
for t in l_text:
    t.set_size=(30)
for t in p_text:
    t.set_size=(20)

plt.axis('equal')
plt.legend()
plt.show()
'''

# make bar chart
'''
plt.bar(np.arange(28), counter)
plt.xlabel(labels)
plt.ylabel('Count')
plt.title('Count of All Types of Proteins')
plt.show()
'''

f.close()
