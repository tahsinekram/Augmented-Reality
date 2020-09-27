import os
import cv2
import numpy as np

import aug


IMG_DIR = "input_images"
VID_DIR = "input_videos"
OUT_DIR = "./"
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

def project_video_on_video(video_name1, video_name2, fps, frame_ids, output_prefix,
                            counter_init):

    video1 = os.path.join(VID_DIR, video_name1)
    video2 = os.path.join(VID_DIR, video_name2)

    image_gen1 = aug.video_frame_generator(video1)
    image_gen2 = aug.video_frame_generator(video2)

    image1 = image_gen1.__next__()
    h1, w1, d1 = image1.shape

    image2 = image_gen2.__next__()
    h2, w2, d2 = image2.shape

    out_path = "output/ar_{}-{}".format(output_prefix[4:], video_name1)
    video_out = mp4_video_writer(out_path, (w1, h1), fps)

    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))
    output_counter = counter_init

    frame_num = 1

    while image1 is not None and image2 is not None:

        print ("Processing frame {}".format(frame_num))

        advert = image2
        src_points = aug.get_corners_list(advert)

        markers = aug.find_markers(image1, template)

        homography = aug.find_four_point_transform(src_points, markers)
        image_out = aug.project_imageA_onto_imageB(advert, image1.copy(), homography)

        frame_id = frame_ids[(output_counter - 1) % 3]

        if frame_num == frame_id:
            out_str = output_prefix + "-{}.png".format(output_counter)
            save_image(out_str, image_out)
            output_counter += 1

        video_out.write(image_out)

        image1 = image_gen1.__next__()
        image2 = image_gen2.__next__()

        frame_num += 1

    video_out.release()


def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(os.path.join(OUT_DIR, filename), image)


def mark_location(image, pt):
    """Draws a dot on the marker center and writes the location as text nearby.

    Args:
        image (numpy.array): Image to draw on
        pt (tuple): (x, y) coordinate of marker center
    """
    color = (0, 50, 255)
    cv2.circle(image, pt, 3, color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "(x:{}, y:{})".format(*pt), (pt[0]+15, pt[1]), font, 0.5, color, 1)

def run():

    video_file = "ps3-4-a.mp4"
    my_video = "my-ad.mp4"  # Place your video in the input_video directory
    frame_ids = [355, 555, 725]
    fps = 40

    project_video_on_video(video_file, my_video, fps, frame_ids, "ps3-6-a", 1)

if __name__ == '__main__':
    run()
