def ds_short_name(ds_name):
    if ds_name == 'bio-diseasome':
        return 'bio-dis'
    if ds_name == 'bio-wormnet':
        return 'bio-worm'
    elif ds_name == 'california':
        return 'ca'
    elif ds_name == 'facebook':
        return 'fb'
    elif ds_name == 'road-minnesota':
        return 'road-m'
    elif ds_name == 'bun_zipper_res3':
        return 'bunny (HR)'
    elif ds_name == 'bun_zipper_res4':
        return 'bunny (LR)'
    elif ds_name == 'drill_shaft_zip':
        return 'drill-shaft'
    elif ds_name == 'regular_sphere1000':
        return 'sphere-mesh'
    elif ds_name == 'catcortex':
        return 'cat-cortex'
    else:
        return ds_name
