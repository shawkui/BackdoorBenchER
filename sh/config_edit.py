import os
import re

'''
A script to edit the config files
'''

base_dir = './config/defense'
for data in ['config', 'cifar10', 'cifar100', 'gtsrb', 'tiny']:
    for reserved_type in ['train', 'reserved', 'reserved_synthetic', 'reserved_brightness', 'reserved_noise', 'reserved_imagenet']:
        # base config file
        target_name = f'{data}.yaml'
        
        # new config file
        new_file_name = f'{data}_{reserved_type}.yaml'

        # search
        for root, dirs, files in os.walk(base_dir):
            for file_name in files:
                if target_name==file_name:
                    file_path = os.path.join(root, file_name)
                    new_file_path = os.path.join(root, new_file_name)
                    
                    # the line to add
                    entry = 'reserved_type'
                    value = reserved_type
                    new_line_text = f'{entry}: {value}'

                    print(f'{file_path} => {new_file_path}')                    
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()

                    with open(new_file_path, 'w', encoding='utf-8') as file:
                        modified=False
                        for line in lines:
                            if entry in line:                        
                                file.write('\n')
                                file.write(new_line_text + '\n')
                                modified = True
                            else:
                                file.write(line)

                        if not modified:
                            file.write('\n')
                            file.write(new_line_text + '\n')
