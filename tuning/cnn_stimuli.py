__author__ = 'bptripp'

from os import listdir, makedirs
from os.path import join, isfile, basename, exists
import numpy as np
from scipy import misc
import string

import matplotlib
import matplotlib.pyplot as plt

def get_image_file_list(source_path, extension, with_path=False):
    if with_path:
        result = [join(source_path, f) for f in listdir(source_path) if isfile(join(source_path, f)) and f.endswith(extension)]
    else:
        result = [f for f in listdir(source_path) if isfile(join(source_path, f)) and f.endswith(extension)]

    return result


def alpha(i):
    if i >= 26**2 or i < 0:
        raise Exception('Can only handle indices from 0 to 675')

    first = i/26
    second = np.mod(i, 26)
    return string.ascii_uppercase[first] + string.ascii_uppercase[second]


def process_lehky_files(source_path, dest_path):
    """
    Reads raw files from Lehky et al. (2011) supplementary material and converts them to a good form
    for AlexNet. Strips the border and embeds in larger gray image.

    :param source_path: Where original images are
    :param dest_path: Where processed images go
    :return:
    """
    border = 2
    full_shape = [256,256,3]
    background = 127
    files = get_image_file_list(source_path, 'ppm')

    if not exists(dest_path):
        makedirs(dest_path)

    for file in files:
        image = misc.imread(join(source_path, file))
        cropped = clean_lehky_image(image, border, background)
        # cropped = image[border:(image.shape[0]-border),border:(image.shape[1]-border),:]
        # cropped = np.maximum(255-cropped, 1) #without this there are various artefacts, don't know why
        #
        # for i in range(cropped.shape[0]):
        #     for j in range(cropped.shape[1]):
        #         distance = np.linalg.norm(cropped[i,j,0] - [background,background,background])
        #         if distance < 15:
        #             cropped[i,j,:] = background

        full = np.zeros(full_shape)
        full[:] = 127

        corner = [full_shape[i]/2 - 1 - cropped.shape[i]/2 for i in range(2)]
        full[corner[0]:corner[0]+cropped.shape[0],corner[1]:corner[1]+cropped.shape[1],:] = cropped
        # plt.imshow(cropped)
        # plt.imshow(full)
        # plt.show()

        misc.imsave(join(dest_path, file[:-4]+'.png'), full)


def clean_lehky_image(image, border, background):
    cropped = image[border:(image.shape[0]-border),border:(image.shape[1]-border),:]
    cropped = np.maximum(255-cropped, 1) #without this there are various artefacts, don't know why
    cropped = 255 - cropped

    for i in range(cropped.shape[0]):
        for j in range(cropped.shape[1]):
            distance = np.linalg.norm(cropped[i,j,0] - [background,background,background])
            if distance < 15:
                cropped[i,j,:] = background

    return cropped


def make_orientations(source_image_file, orientations, dest_path):
    if not exists(dest_path):
        makedirs(dest_path)

    source_name = basename(source_image_file)[:-4]

    source_image = misc.imread(source_image_file)
    scale = 100. / np.max(source_image.shape)
    source_image = misc.imresize(source_image, scale)

    full_dim = 256
    ccr_dim = int(2*np.ceil(full_dim/2*2**.5)) # cover corners when rotated
    # ccr_shape = [ccr_dim,ccr_dim]

    background_colour = source_image[0,0,:]
    big_image = np.tile(background_colour, [ccr_dim,ccr_dim,1])
    corner = [ccr_dim/2-source_image.shape[0]/2, ccr_dim/2-source_image.shape[1]/2]
    ss = source_image.shape
    big_image[corner[0]:corner[0]+ss[0],corner[1]:corner[1]+ss[1],:] = source_image

    # plt.imshow(background)
    # plt.show()
    # bg = np.round(np.mean(source_image[0:3,0:3,:]))
    # print(bg)
    # print(source_image[0:3,0:3,:])
    # buffered = np.

    for orientation in orientations:
        rotated = misc.imrotate(big_image, orientation, interp='bilinear')
        crop = (ccr_dim - full_dim)/2
        cropped = rotated[crop:-crop,crop:-crop,:]
        # plt.imshow(cropped)
        # plt.show()
        misc.imsave(join(dest_path, source_name + alpha(int(orientation)) + '.png'), cropped)


def add_borders(source_path, dest_path, scale=.2, dim=256, extension='png'):
    if not exists(dest_path):
        makedirs(dest_path)

    files = get_image_file_list(source_path, extension)

    for file in files:
        image = misc.imread(join(source_path, file), mode='RGB')
        image = misc.imresize(image, scale)

        full = np.zeros((dim,dim,3), dtype=image.dtype)
        full[:] = 255

        corner = [dim/2 - 1 - image.shape[i]/2 for i in range(2)]
        full[corner[0]:corner[0]+image.shape[0],corner[1]:corner[1]+image.shape[1],:] = image

        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.imshow(image)
        # plt.subplot(2,1,2)
        # plt.imshow(full)
        # plt.show()

        misc.imsave(join(dest_path, file), full)



def make_sizes(source_image_file, scales, dest_path):
    if not exists(dest_path):
        makedirs(dest_path)

    # print(source_image_file)
    source_name = basename(source_image_file)[:-4]
    source_image = misc.imread(source_image_file)
    background_colour = source_image[0,0,:]
    dim = 256

    for i in range(len(scales)):
        result = np.tile(background_colour, [dim,dim,1])
        scaled = misc.imresize(source_image, scales[i])

        if scaled.shape[0] > dim:
            trim = int((scaled.shape[0]-dim+1)/2)
            scaled = scaled[trim:-trim,:,:]

        if scaled.shape[1] > dim:
            trim = int((scaled.shape[1]-dim+1)/2)
            scaled = scaled[:,trim:-trim,:]

        # print(scales[i])
        # print(scaled.shape)
        c = int(np.floor((dim-scaled.shape[0])/2)), int(np.floor((dim-scaled.shape[1])/2)) #corner
        result[c[0]:c[0]+scaled.shape[0],c[1]:c[1]+scaled.shape[1]] = scaled

        misc.imsave(join(dest_path, source_name + alpha(i) + '.png'), result)
        # plt.imshow(result)
        # plt.show()


def make_positions(source_image_file, scale, offsets, dest_path):
    if not exists(dest_path):
        makedirs(dest_path)

    source_name = basename(source_image_file)[:-4]
    source = misc.imread(source_image_file)
    source = misc.imresize(source, scale)
    background_colour = source[0,0,:]
    dim = 256

    for i in range(len(offsets)):
        result = np.tile(background_colour, [dim,dim,1])

        top = (dim-source.shape[0])/2
        bottom = top+source.shape[0]
        left = (dim-source.shape[1])/2+offsets[i]
        right = left + source.shape[1]

        section = source[:,np.maximum(0,-left):-1-np.maximum(0,right-dim),:]
        result[top:bottom,np.maximum(0,left):np.minimum(dim-1,right-1),:] = section

        misc.imsave(join(dest_path, source_name + alpha(i) + '.png'), result)
        # plt.imshow(result)
        # plt.show()


def make_positions_schwartz(source_image_file, offset, dest_path):
    if not exists(dest_path):
        makedirs(dest_path)

    source_name = basename(source_image_file)[:-4]
    source = misc.imread(source_image_file)
    background_colour = source[0,0,:]
    dim = 256

    hor_offsets = [-offset, offset, 0, 0, 0]
    ver_offsets = [0, 0, -offset, offset, 0]

    for i in range(len(hor_offsets)):
        result = np.tile(background_colour, [dim,dim,1])

        top = (dim-source.shape[0])/2+ver_offsets[i]
        bottom = top+source.shape[0]
        left = (dim-source.shape[1])/2+hor_offsets[i]
        right = left + source.shape[1]

        section = source[:,np.maximum(0,-left):-1-np.maximum(0,right-dim),:]
        result[top:bottom,np.maximum(0,left):np.minimum(dim-1,right-1),:] = section

        misc.imsave(join(dest_path, source_name + alpha(i) + '.png'), result)
        # plt.imshow(result)
        # plt.show()


def make_occlusions(dest_path, shape_colour=[255,255,255], motion=False):
    def make_background():
        return 1 + 254*np.tile(np.random.randint(0, 2, [256,256,1]), [1,1,3])

    # can't see how to extract pixels from image on mac, so drawing lines manually
    def draw_line(image, p1, p2, width):
        left = int(max(0, min(p1[1]-width, p2[1]-width)))
        right = int(min(image.shape[1]-1, max(p1[1]+width, p2[1]+width)))
        top = int(max(0, min(p1[0]-width, p2[0]-width)))
        bottom = int(min(image.shape[1]-1, max(p1[0]+width, p2[0]+width)))

        a = p1[0]-p2[0]
        b = p2[1]-p1[1]
        c = -a*p1[1] - b*p1[0]

        for i in range(top, bottom):
            for j in range(left, right):
                if p1[0] == p2[0]: #horizontal
                    d = abs(i-p1[0])
                elif p1[1] == p2[1]: #vertical
                    d = abs(j-p1[1])
                else:
                    d = abs(a*j + b*i + c) / (a**2 + b**2)**.5

                val = 255
                if d < width:
                    # image[i,j,:] = 255
                    image[i,j,:] = shape_colour
                elif d - width < 1:
                    image[i,j,:] = (d-width)*image[i,j,:] + (1-d+width)*np.array(shape_colour, dtype=int) #[val,val,val]

    def draw_contour(image, x, y, width):
        for i in range(len(x)-1):
            draw_line(image, (x[i],y[i]), (x[i+1],y[i+1]), width)

    def occlude(image, p):
        block_dim = 8
        original_image = image.copy()
        for i in range(image.shape[0]/block_dim):
            for j in range(image.shape[1]/block_dim):
                if np.random.rand() < p:
                    if not motion:
                        image[block_dim*i:block_dim*(i+1), block_dim*j:block_dim*(j+1), :] = 255
                    else:
                        # simulate motion of ~1.5 degrees diagonally by moving down and right 20 pixels
                        # simulate transience of occlusion at each point with transparency
                        opacity = .05
                        for k in range(20):
                            top = block_dim*i+k
                            bottom = min(block_dim*(i+1)+k,image.shape[0])
                            left = block_dim*j+k
                            right = min(block_dim*(j+1)+k, image.shape[1])
                            change = opacity * (255 - original_image[top:bottom, left:right, :])
                            image[top:bottom, left:right, :] = image[top:bottom, left:right, :] + change
                            # image[top:bottom, left:right, :] \
                            #     = (1-opacity) * image[top:bottom, left:right, :] \
                            #     + opacity * 255

    def save_occlusions(name, x, y, line_width):
        d = join(dest_path, name)
        if not exists(d):
            makedirs(d)

        # x, y: lists of coordinates of shape outline to plot
        percent_occlusion = [0, 20, 50, 90, 100]
        for p in percent_occlusion:
            for rep in range(10):
                image = make_background()
                draw_contour(image, x, y, line_width)
                occlude(image, p/100.)
                # the 99 is a hack to make the files load in the expected order (not alphabetically)
                misc.imsave(join(dest_path, name, name + str(np.minimum(p,99)) + '-' + str(rep) + '.png'), 255-image)
                # plt.imshow(image)
                # plt.show()

    angle = np.linspace(0, 2*np.pi)
    save_occlusions('circle', 128+30*np.cos(angle), 128+30*np.sin(angle), 2)
    angle = np.linspace(0, 2*np.pi, 13)
    radii = [30-15*np.mod(i,2) for i in range(len(angle))]
    save_occlusions('star', 128+radii*np.cos(angle), 128+radii*np.sin(angle), 2)
    a,b = 108,148
    save_occlusions('square', [a,b,b,a,a], [a,a,b,b,a], 2)
    save_occlusions('triangle', 128+np.array([18,18,-18,18]), 128+np.array([-18,18,0,-18]), 2)
    save_occlusions('strange', 128+np.array([20,20,10,10,-20,-20,-10,-10,0,0,10,10,20]), 128+np.array([-30,20,20,10,10,0,0,-20,-20,0,0,-30,-30]), 2)

    # circle_image = make_background()
    # angle = np.linspace(0, 2*np.pi)
    # draw_contour(circle_image, 128+20*np.cos(angle), 128+20*np.sin(angle), 3)


def make_clutters(source_path, dest_path):
    if not exists(dest_path):
        makedirs(dest_path)
        makedirs(join(dest_path+'/top'))
        makedirs(join(dest_path+'/bottom'))
        makedirs(join(dest_path+'/pair'))

    full_shape = [256,256,3]
    background_colour = 127
    files = get_image_file_list(source_path, 'ppm')

    images = []
    for file in files:
        image = misc.imread(join(source_path, file))
        images.append(clean_lehky_image(image, 2, background_colour))

    tops = images[:4]
    bottoms = images[4:]

    corner_top_image = [66,99] #top left corner of top image
    corner_bottom_image = [132,99] # top left corner of bottom image

    def get_background():
        background = np.zeros(full_shape)
        background[:] = background_colour
        return background

    def add_image(background, image, corner):
        background[corner[0]:corner[0]+image.shape[0], corner[1]:corner[1]+image.shape[1] ,:] = image

    for i in range(len(tops)):
        full = get_background()
        add_image(full, tops[i], corner_top_image)
        misc.imsave(join(dest_path+'/top', 'top' + str(i) + '.png'), full)

    for i in range(len(bottoms)):
        full = get_background()
        add_image(full, bottoms[i], corner_bottom_image)
        misc.imsave(join(dest_path+'/bottom', 'bottom' + str(i) + '.png'), full)

    # all pairs
    for i in range(len(tops)):
        for j in range(len(bottoms)):
            full = get_background()
            add_image(full, tops[i], corner_top_image)
            add_image(full, bottoms[j], corner_bottom_image)

            misc.imsave(join(dest_path+'/pair', 'pair' + str(i) + '-' + str(j) + '.png'), full)
            # full = np.zeros(full_shape)
            # full[:] = 127
            #
            # full[corner_top_image[0]:corner_top_image[0]+images[i].shape[0],corner_top_image[1]:corner_top_image[1]+images[i].shape[1],:] = images[i]
            # full[corner_bottom_image[0]:corner_bottom_image[0]+images[j].shape[0],corner_bottom_image[1]:corner_bottom_image[1]+images[j].shape[1],:] = images[j]
            #
            # plt.imshow(full)
            # plt.show()

def make_3d(source_dir, dest_dir):
    # crop and resize images appropriately
    target_shape = [256,256,3]
    source_files = get_image_file_list(source_dir, 'jpg')
    for source_file in source_files:
        im = misc.imread(join(source_dir, source_file))
        excess = im.shape[1] - im.shape[0]
        cropped = im[:,excess/2:-excess/2]
        resized = misc.imresize(cropped, target_shape)
        # print(im.shape)
        # print(cropped.shape)
        # print(resized.shape)
        misc.imsave(join(dest_dir, source_file), resized)

    # source = misc.imread(source_image_file)



if __name__ == '__main__':
    # add_borders('./source-images/simplification/from', './images/simplification/from')
    # add_borders('./source-images/simplification/to', './images/simplification/to')

    # # print(get_image_file_list('./source-images/lehky', 'ppm'))
    #
    # print('Cleaning Lehky et al. images ...')
    # process_lehky_files('./source-images/lehky',
    #                     './images/lehky-processed')
    #
    # print('Making orientation stimuli ... ')
    # make_orientations('./source-images/banana.png',
    #                   np.linspace(0, 360, 91),
    #                   './images/banana-rotations')
    #
    # make_orientations('./source-images/shoe.png',
    #                   np.linspace(0, 360, 91),
    #                   './images/shoe-rotations')
    #
    # make_orientations('./source-images/corolla.png',
    #                   np.linspace(0, 360, 91),
    #                   './images/corolla-rotations')
    #
    # make_orientations('./source-images/swiss_knife.jpg',
    #                   np.linspace(0, 360, 91),
    #                   './images/swiss-knife-rotations')
    #
    # make_orientations('./source-images/staple.png',
    #                   np.linspace(0, 360, 91),
    #                   './images/staple-rotations')
    #
    # print('Making clutter stimuli ... ')
    # make_clutters('./source-images/clutter',
    #               './images/clutter')
    #
    print('Making occlusion stimuli ... ')
    make_occlusions('./images/occlusions-black')
    # make_occlusions('./images/occlusions-red', shape_colour=[0,255,255])
    # Note in Kovacs et al, the shape was typically presented for 500ms, the occlusion
    # pattern moved at 3deg/s, and shapes covered an area of about 10deg^2, so the
    # occlusion pattern moved about half the shape height (diagonally down)
    # make_occlusions('./images/occlusions-moving', motion=True)

    #
    # print('Making size tuning stimuli ... ')
    # schwartz_scales = [13**.5/28**.5, 1., 50**.5/28**.5]
    # make_sizes('./source-images/schwartz/f1.png',
    #            schwartz_scales,
    #            './images/scales/f1')
    # make_sizes('./source-images/schwartz/f2.png',
    #            schwartz_scales,
    #            './images/scales/f2')
    # make_sizes('./source-images/schwartz/f3.png',
    #            schwartz_scales,
    #            './images/scales/f3')
    # make_sizes('./source-images/schwartz/f4.png',
    #            schwartz_scales,
    #            './images/scales/f4')
    # make_sizes('./source-images/schwartz/f5.png',
    #            schwartz_scales,
    #            './images/scales/f5')
    # make_sizes('./source-images/schwartz/f6.png',
    #            schwartz_scales,
    #            './images/scales/f6')
    #
    # scales = np.logspace(np.log10(.05), np.log10(1.2), 45)
    # make_sizes('./source-images/corolla.png',
    #            scales,
    #            './images/scales/corolla')
    #
    # make_sizes('./source-images/shoe.png',
    #            scales,
    #            './images/scales/shoe')
    #
    # make_sizes('./source-images/banana.png',
    #            scales,
    #            './images/scales/banana')
    #
    # print('Making position tuning stimuli ... ')
    # offsets = np.linspace(-75, 75, 150/5+1, dtype=int)
    # make_positions('./source-images/shoe.png', .4,
    #            offsets,
    #            './images/positions/shoe')
    #
    # make_positions('./source-images/banana.png', .4,
    #            offsets,
    #            './images/positions/banana')
    #
    # make_positions('./source-images/corolla.png', .4,
    #            offsets,
    #            './images/positions/corolla')
    #
    # make_positions('./source-images/staple.png', .4,
    #            offsets,
    #            './images/positions/staple')
    #
    # # From Schwartz et al., 5 degrees up, down, left right with stimulus 28**.5=5.3 degrees wide
    # # our stimuli ~2/3 * 56 pixels = 37, so we want shifts of 35 pixels
    # schwartz_offset = 35
    # make_positions_schwartz('./source-images/schwartz/f1.png',
    #                         schwartz_offset,
    #                         './images/positions/f1')
    # make_positions_schwartz('./source-images/schwartz/f2.png',
    #                         schwartz_offset,
    #                         './images/positions/f2')
    # make_positions_schwartz('./source-images/schwartz/f3.png',
    #                         schwartz_offset,
    #                         './images/positions/f3')
    # make_positions_schwartz('./source-images/schwartz/f4.png',
    #                         schwartz_offset,
    #                         './images/positions/f4')
    # make_positions_schwartz('./source-images/schwartz/f5.png',
    #                         schwartz_offset,
    #                         './images/positions/f5')
    # make_positions_schwartz('./source-images/schwartz/f6.png',
    #                         schwartz_offset,
    #                         './images/positions/f6')

    # make_3d('/Users/bptripp/code/salman-IT/salman/images/scooter',
    #         '/Users/bptripp/code/salman-IT/salman/images/scooter-cropped')
    # make_3d('/Users/bptripp/code/salman-IT/salman/images/staple',
    #         '/Users/bptripp/code/salman-IT/salman/images/staple-cropped')
    # make_3d('/Users/bptripp/code/salman-IT/salman/images/head',
    #         '/Users/bptripp/code/salman-IT/salman/images/head-cropped')
