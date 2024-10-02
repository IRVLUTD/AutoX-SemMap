# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Factory method for easily getting imdbs by name."""

__sets = {}

from .osd_object import OSDObject
from .ocid_object import OCIDObject


# OSD object dataset
for split in ['test']:
    name = 'osd_object_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            OSDObject(split))

# OCID object dataset
for split in ['test']:
    name = 'ocid_object_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            OCIDObject(split))

def get_dataset(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_datasets():
    """List all registered imdbs."""
    return __sets.keys()
