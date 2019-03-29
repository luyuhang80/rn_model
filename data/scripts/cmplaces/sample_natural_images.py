import os
import random
import shutil
import math

root_dir = '/mnt/db/data/cmplaces'
src_dir = '/mnt/db/data/cmplaces/original/natural'
categories_path = '/mnt/db/data/cmplaces/labels/categories.txt'
series = 'natural100k'
out_dir = os.path.join(root_dir, series)
os.makedirs(out_dir, exist_ok=True)

n_classes = 205
images_per_class = 500
val_portion = 0.1
test_portion = 0.1

train_list = []
val_list = []
test_list = []

with open(categories_path, 'r') as f:
    categories = [line[:line.find(' ')] for line in f.readlines()]

idx = 1
for label in range(n_classes):
    cat = categories[label]
    cat_path = os.path.join(src_dir, cat)
    files = [f for f in os.listdir(cat_path) if f.endswith('.jpg')]
    print("category %d %s originally has %d images" % (label, categories[label], len(files)))
    files = random.sample(files, images_per_class)

    new_list = []
    for file in files:
        old_path = os.path.join(cat_path, file)
        new_name = '%s_%08d_%d.jpg' % (series, idx, label)
        new_path = os.path.join(out_dir, new_name)
        new_list.append(new_name)
        shutil.copy(old_path, new_path)
        idx += 1

    n_val = int(math.ceil(images_per_class * val_portion))
    n_test = int(math.ceil(images_per_class * test_portion))
    n_train = int(math.ceil(images_per_class * (1.0 - val_portion - test_portion)))
    val_list += new_list[:n_val]
    test_list += new_list[n_val:n_val+n_test]
    train_list += new_list[-n_train:]

    print('class %d: %d, %d train, %d val, %d test' % (label, images_per_class, n_train, n_val, n_test))

train_list_path = os.path.join(root_dir, '%s_%s.txt' % (series, 'train'))
val_list_path = os.path.join(root_dir, '%s_%s.txt' % (series, 'val'))
test_list_path = os.path.join(root_dir, '%s_%s.txt' % (series, 'test'))

# with open(train_list_path, 'w') as f: f.write('\n'.join(train_list))
# with open(val_list_path, 'w') as f: f.write('\n'.join(val_list))
# with open(test_list_path, 'w') as f: f.write('\n'.join(test_list))

n_train = len(train_list)
n_val = len(val_list)
n_test = len(test_list)
n = n_train + n_val + n_test
print('total %d images, %d train, %d val, %d test' % (n, n_train, n_val, n_test))



