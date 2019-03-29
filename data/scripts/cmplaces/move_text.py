import os
import pathlib
import math
import shutil

dataset_path = '/mnt/db/data/cmplaces'
series = 'text'
output_path = os.path.join(dataset_path, series)
pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) # recursively makes directories if they don't already exist
categories_path = '/mnt/db/data/cmplaces/labels/categories.txt'
old_train_list_path = '/mnt/db/data/cmplaces/labels/text_train.txt'
old_test_list_path = '/mnt/db/data/cmplaces/labels/text_test.txt'
n_classes = 205
val_portion = 0.1
test_portion = 0.1

with open(old_train_list_path, 'r') as f:
    train_list = [line.strip() for line in f.readlines()]
train_list = [(l[:l.find(' ')], int(l[l.find(' ') + 1:])) for l in train_list]

with open(old_test_list_path, 'r') as f:
    test_list = [line.strip() for line in f.readlines()]
test_list = [(l[:l.find(' ')], int(l[l.find(' ') + 1:])) for l in test_list]

combined_list = train_list + test_list
list_by_classes = [[] for i in range(n_classes)]
for file, label in combined_list:
    file = 'original' + file[4:]
    file_path = os.path.join(dataset_path, file)
    list_by_classes[label].append(file_path)

serial = 1
train_list = []
val_list = []
test_list = []

serial = 1
train_list = []
val_list = []
test_list = []

for i in range(n_classes):
    class_list = list_by_classes[i]
    n = len(class_list)
    n_test = int(math.ceil(n * test_portion))
    n_val = int(math.ceil(n * val_portion))
    n_train = n - n_val - n_test
    if n_train <= 0: n_train = n
    print('class %d: %d, %d train, %d val, %d test' % (i, n, n_train, n_val, n_test))

    new_list = []
    for file_path in class_list:
        new_name = '%s_%08d_%d.txt' % (series, serial, i)
        serial += 1
        new_list.append(new_name)
        new_path = os.path.join(output_path, new_name)
        shutil.copy(file_path, new_path)

    test_list += new_list[:n_test]
    if n > n_test + n_val:
        val_list += new_list[n_test:n_test+n_val]
    else:
        val_list += new_list[:n_val]
    train_list += new_list[-n_train:]


train_list_path = os.path.join(dataset_path, '%s_%s.txt' % (series, 'train'))
val_list_path = os.path.join(dataset_path, '%s_%s.txt' % (series, 'val'))
test_list_path = os.path.join(dataset_path, '%s_%s.txt' % (series, 'test'))

with open(train_list_path, 'w') as f: f.write('\n'.join(train_list))
with open(val_list_path, 'w') as f: f.write('\n'.join(val_list))
with open(test_list_path, 'w') as f: f.write('\n'.join(test_list))

n_train = len(train_list)
n_val = len(val_list)
n_test = len(test_list)
n = n_train + n_val + n_test
print('total %d samples, %d train, %d val, %d test' % (n, n_train, n_val, n_test))