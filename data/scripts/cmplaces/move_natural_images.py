import os
from subprocess import call

src_path = '/mnt/db/data/cmplaces/original/natural_letters'
out_path = '/mnt/db/data/cmplaces/original/natural'
os.makedirs(out_path, exist_ok=True)

letters = os.listdir(src_path)
letters = sorted(letters)
for l in letters:
    letter_path = os.path.join(src_path, l)
    dirs = os.listdir(letter_path)
    for d in dirs:
        subdirs = os.listdir(os.path.join(src_path, l, d))
        subdirs = [sd for sd in subdirs if os.path.isdir(os.path.join(src_path, l, d, sd))]
        if len(subdirs) > 0:
            for sd in subdirs:
                new_dir_name = d + '_' + sd
                old_dir_path = os.path.join(src_path, l, d, sd)
                new_dir_path = os.path.join(out_path, new_dir_name)
                os.makedirs(new_dir_path, exist_ok=True)
                command = ['mv', old_dir_path + '/*', new_dir_path + '/']
                print(' '.join(command))
                os.system(' '.join(command))
        else:
            old_dir_path = os.path.join(src_path, l, d)
            new_dir_path = os.path.join(out_path, d)
            command = ['mv', old_dir_path, new_dir_path]
            print(' '.join(command))
            os.system(' '.join(command))