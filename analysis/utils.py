import os


def remove_extensions(path):
    basename = os.path.basename(path)
    if '+' in basename:
        basename = basename[:basename.index('+')]
    if '.' in basename:
        basename = basename[:basename.index('.')]
    return basename


def _output_filename(input_path, name, inherit, extension):
    if inherit:
        basename = remove_extensions(input_path)
        filename = f'{basename}+{name}.{extension}'
    else:
        filename = f'{name}.{extension}'
    return os.path.join(os.path.dirname(input_path), filename)


def np_output_filename(input_path, name, inherit):
    return _output_filename(input_path, name, inherit, 'npy')


def annotate_vline(value, text, left=True, **kwargs):
    import matplotlib.pyplot as plt
    plt.axvline(value, **kwargs)
    xoff, align = (-15, 'left') if left else (15, 'right')
    plt.annotate(
            text,
            xy=(value, 1),
            xytext=(xoff, 15),
            xycoords=('data', 'axes fraction'),
            textcoords='offset points',
            horizontalalignment=align,
            verticalalignment='center',
            arrowprops=dict(
                    arrowstyle='-|>',
                    fc='black',
                    shrinkA=0,
                    shrinkB=0,
                    connectionstyle='angle,angleA=0,angleB=90,rad=10'))
