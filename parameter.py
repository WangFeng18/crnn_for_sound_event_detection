import os
# constant
n_classes = 16
hop_length = 20
id_name = ['alarms_and_sirens', 'baby_crying', 'bird_singing',
        'bus', 'cat_meowing', 'crowd_applause',
        'crowd_cheering', 'dog_barking', 'footsteps',
        'glass_smash', 'gun_shot', 'horsewalk',
        'mixer', 'motorcycle', 'rain', 'thunder',
        ]
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
estimate_dir = 'data/TUT_SED/TUT-SED-synthetic-2016/estimate/'
audio_dir = 'data/TUT_SED/TUT-SED-synthetic-2016/audio/'
audio_template = 'TUT-SED-synthetic-2016-mix-{}.wav'
feat_template = 'TUT-SED-synthetic-2016-mix-{}.cpickle'
meta_template = 'TUT-SED-synthetic-2016-annot-{}.txt'
train = range(40,100)
evaluate = range(20)

# variable
# segment_length = 20
