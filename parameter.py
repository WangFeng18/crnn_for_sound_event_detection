import os
n_classes = 16
hop_length = 20
segment_length = 20
name_id = {'alarms_and_sirens':0,\
        'baby_crying':1,\
        'bird_singing':2,
        'bus':3,
        'cat_meowing':4,
        'crowd_applause':5,
        'crowd_cheering':6,
        'dog_barking':7,
        'footsteps':8,
        'glass_smash':9,
        'gun_shot':10,
        'horsewalk':11,
        'mixer':12,
        'motorcycle':13,
        'rain':14,
        'thunder':15,
        }
feature_dir = 'data/TUT_SED/TUT-SED-synthetic-2016/features/'
meta_dir = 'data/TUT_SED/TUT-SED-synthetic-2016/meta/'
train = [(meta_dir+'TUT-SED-synthetic-2016-annot-{}.txt'.format(i), feature_dir+'TUT-SED-synthetic-2016-mix-{}.cpickle'.format(i)) for i in range(40,100)]
evaluate = [(meta_dir+'TUT-SED-synthetic-2016-annot-{}.txt'.format(i),feature_dir+'TUT-SED-synthetic-2016-mix-{}.cpickle'.format(i)) for i in range(20)]
