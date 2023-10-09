import argparse
import os
import tkinter
import random
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import distinctipy
from inhand_utils import read_dict, write_dict, get_new_basename
#random.seed(3)
random.seed(2)

width = 1440
height = 1080
circule_radius = 2

resize = True
resize_scale = 1.8

class Annotate():
    def __init__(self, pairs_path, detections_path, image_dir, cluster_dir,
                 output_dir, save_default, shuffle):
        self.pairs_path = pairs_path
        self.detections = read_dict(detections_path)
        self.image_dir = image_dir
        self.clsuter_dir = cluster_dir
        self.output_dir = output_dir
        self.save_default = save_default
        self.shuffle = shuffle

        self.should_quit = None
        self.should_delete = None
        self.should_save = None
        self.pair_index = None
        self.prev_index = None
        self.num_pairs = None

        self.curr_pair = None
        self.curr_associations_dict = None

        self.label_mode = None
        self.fruitlet_num = None

    def annotate(self):
        # set up the gui
        window = tkinter.Tk()
        window.bind("<Key>", self.event_action)
        window.bind("<Button-1>", self.event_action_click)
        # get the pairs of jsons
        #already sorted
        pairs = read_dict(self.pairs_path)

        if self.shuffle:
            random.shuffle(pairs)

        self.should_quit = False
        self.pair_index = 0
        self.prev_index = None
        self.num_pairs = len(pairs)
        self.label_mode = False

        while not self.should_quit:
            self.should_save = self.save_default
            self.should_delete = False

            #these are annotations to be saved
            tmp_path_0, tmp_path_1 = pairs[self.pair_index]

            basename_0 = get_new_basename(tmp_path_0).split('.png')[0]
            basename_1 = get_new_basename(tmp_path_1).split('.png')[0]
            file_id = '_'.join([basename_0, basename_1])

            image_path_0 = os.path.join(self.image_dir, basename_0 + '.png')
            image_path_1 = os.path.join(self.image_dir, basename_1 + '.png')

            annotations_0 = self.detections[os.path.basename(image_path_0)]
            annotations_1 = self.detections[os.path.basename(image_path_1)]

            clusters_path_0 = os.path.join(self.clsuter_dir, basename_0 + '.json')
            clusters_path_1 = os.path.join(self.clsuter_dir, basename_1 + '.json')
            clusters_0 = read_dict(clusters_path_0)
            clusters_1 = read_dict(clusters_path_1)
            colors_0 = distinctipy.get_colors(clusters_0['num_clusters'], rng=0)
            colors_1 = distinctipy.get_colors(clusters_1['num_clusters'], rng=0)

            associations_file = os.path.join(self.output_dir, file_id + '.json')

            if self.pair_index == self.prev_index:
                pass
            elif os.path.exists(associations_file):
                self.curr_associations_dict = read_dict(associations_file)
            else:
                self.curr_associations_dict = {}
                self.curr_associations_dict['image_0'] = image_path_0
                self.curr_associations_dict['image_1'] = image_path_1
                self.curr_associations_dict['annotations_0'] = []
                self.curr_associations_dict['annotations_1'] = []

                for box in annotations_0:
                    x0, y0, x1, y1, score = box

                    det = {}
                    det['x0'] = x0
                    det['y0'] = y0
                    det['x1'] = x1
                    det['y1'] = y1
                    det['score'] = score
                    det['assoc_id'] = -1

                    self.curr_associations_dict["annotations_0"].append(det)

                for box in annotations_1:
                    x0, y0, x1, y1, score = box

                    det = {}
                    det['x0'] = x0
                    det['y0'] = y0
                    det['x1'] = x1
                    det['y1'] = y1
                    det['score'] = score
                    det['assoc_id'] = -1

                    self.curr_associations_dict["annotations_1"].append(det)
            
            self.prev_index = self.pair_index

            window.title(file_id)
            picture_0 = Image.open(image_path_0)
            picture_1 = Image.open(image_path_1)

            picture = Image.new('RGB', (width*2, height))
            picture.paste(picture_0, (0,0))
            picture.paste(picture_1, (width,0))

            if resize:
                picture = picture.resize((2*int(width//resize_scale), int(height//resize_scale)))

            picture_draw = ImageDraw.Draw(picture)
            
            is_valid = True
            valid_set = set()
            label_infos = []
            for det_ind in range(len(self.curr_associations_dict['annotations_0'])):
                det = self.curr_associations_dict['annotations_0'][det_ind]

                y0 = int(det['y0'])
                x0 = int(det['x0'])
                y1 = int(det['y1'])
                x1 = int(det['x1'])
                
                if resize:
                    x0 = x0 // resize_scale
                    y0 = y0 // resize_scale
                    x1 = x1 // resize_scale
                    y1 = y1 // resize_scale

                pts = [(x0, y0), (x1, y1)]

                if det['assoc_id'] >= 0:
                    color = 'red'
                elif det['assoc_id'] == -1:
                    #color = "cyan"

                    cluster_num = clusters_0["fruitlet_clusters"][str(det_ind)] 
                    if cluster_num  == 'too_small_area':
                        color = 'black'
                    elif cluster_num == 'bad_disparity':
                        color = (79,79,47)
                    elif cluster_num == 'unassigned':
                        color = 'white'
                    else:
                        cluster_color = colors_0[cluster_num]

                        color = cluster_color
                        color = (int(255*color[0]), int(255*color[1]), int(255*color[2]))
                else:
                    color = "purple"

                picture_draw.rectangle(pts, outline =color)

                mid_x = int((x0 + x1)/2)
                mid_y = int((y0 + y1)/2)

                if (det['assoc_id'] >= 0) and (det['assoc_id'] in valid_set):
                    is_valid = False
                
                valid_set.add(det['assoc_id'])

                picture_draw.ellipse([(mid_x - circule_radius, mid_y - circule_radius), (mid_x + circule_radius, mid_y + circule_radius)], fill='purple')
                if det['assoc_id'] != -1:
                    label_infos.append([mid_x, mid_y, det['assoc_id']])

            valid_set = set()
            for det_ind in range(len(self.curr_associations_dict['annotations_1'])):
                det = self.curr_associations_dict['annotations_1'][det_ind]

                y0 = int(det['y0'])
                x0 = int(det['x0']) + width
                y1 = int(det['y1'])
                x1 = int(det['x1']) + width
                
                if resize:
                    x0 = x0 // resize_scale
                    y0 = y0 // resize_scale
                    x1 = x1 // resize_scale
                    y1 = y1 // resize_scale

                pts = [(x0, y0), (x1, y1)]

                
                
                if det['assoc_id'] >= 0:
                    color = 'red'
                elif det['assoc_id'] == -1:
                    #color = "cyan"

                    cluster_num = clusters_1["fruitlet_clusters"][str(det_ind)] 
                    if cluster_num  == 'too_small_area':
                        color = 'black'
                    elif cluster_num == 'bad_disparity':
                        color = (79,79,47)
                    elif cluster_num == 'unassigned':
                        color = 'white'
                    else:
                        cluster_color = colors_1[cluster_num]

                        color = cluster_color
                        color = (int(255*color[0]), int(255*color[1]), int(255*color[2]))
                else:
                    color = "purple"

                picture_draw.rectangle(pts, outline =color)

                mid_x = int((x0 + x1)/2)
                mid_y = int((y0 + y1)/2)
                
                if (det['assoc_id'] >= 0) and (det['assoc_id'] in valid_set):
                    is_valid = False
                
                valid_set.add(det['assoc_id'])

                picture_draw.ellipse([(mid_x - circule_radius, mid_y - circule_radius), (mid_x + circule_radius, mid_y + circule_radius)], fill='purple')
                if det['assoc_id'] != -1:
                    label_infos.append([mid_x, mid_y, det['assoc_id']])

            tk_picture = ImageTk.PhotoImage(picture)
            picture_width = picture.size[0]
            picture_height = picture.size[1]
            window.geometry("{}x{}+100+100".format(picture_width, picture_height))
            image_widget = tkinter.Label(window, image=tk_picture)
            image_widget.place(x=0, y=0, width=picture_width, height=picture_height)

            if is_valid:
                label_string = 'valid ' + str(file_id)
                label_colour = 'green'
            else:
                label_string = 'invalid ' + str(file_id)
                label_colour = 'red'

            label_text = tkinter.Label(window, text=label_string, font=("Helvetica", 22), fg=label_colour)
            label_text.place(anchor = tkinter.NW, x = 0, y = 0)

            for label_info in label_infos:
                mid_x, mid_y, assoc_id = label_info
                num_text = tkinter.Label(window, text=str(assoc_id), font=("Helvetica", 8))
                num_text.place(x=mid_x, y=mid_y+10)

            # wait for events
            if resize:
                w_txt = str(2*int(width//resize_scale))
                h_txt = str(int(height//resize_scale))
                window.geometry(w_txt + "x" + h_txt)
            window.mainloop()

            if self.should_quit:
                continue

            assert not (self.should_save and self.should_delete)

            if self.should_save:
                write_dict(associations_file, self.curr_associations_dict)

            if self.should_delete:
                if os.path.exists(associations_file):
                    os.remove(associations_file)

    def event_action(self, event):
        character = event.char
        
        if character == 'q':
            self.should_quit = True
            event.widget.quit()
        elif character == 's':
            self.should_save = True
            self.should_delete = False
            self.prev_index = None
            event.widget.quit()
        elif character == 'b':
            self.should_save = False
            self.should_delete = True
            self.prev_index = None
            self.label_mode = False
            self.fruitlet_num = None
            event.widget.quit()
        elif character in ['`', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            self.label_mode = True
            if character == '`':
                self.fruitlet_num = -2
            else:
                self.fruitlet_num = int(character)
        elif character in ['x']:
            self.label_mode = True
            if self.fruitlet_num is None:
                self.fruitlet_num = 0
            else:
                self.fruitlet_num += 1
        elif character in ['p']:
            self.label_mode = True
            self.fruitlet_num = -1
        elif character == 'c':
            self.label_mode = False
            self.fruitlet_num = None
        elif character == 'k':
            self.label_mode = False
            self.fruitlet_num = None
            self.prev_index = None
            event.widget.quit()
        elif character == 'a':
            if self.pair_index > 0:
                self.label_mode = False
                self.fruitlet_num = None
                self.pair_index -= 1
                event.widget.quit()
        elif character == 'd':
            if self.pair_index < self.num_pairs - 1:
                self.label_mode = False
                self.fruitlet_num = None
                self.pair_index += 1
                event.widget.quit()

    def event_action_click(self, event):
        if not self.label_mode:
            return

        x = event.x
        y = event.y

        if resize:
            x = x * resize_scale
            y = y * resize_scale

        if x >= width:
            annotations_dict = self.curr_associations_dict['annotations_1']
            x = x - width
        else:
            annotations_dict = self.curr_associations_dict['annotations_0']

        min_dist = None
        min_det = None
        for det in annotations_dict:
            y0 = int(det['y0'])
            x0 = int(det['x0'])
            y1 = int(det['y1'])
            x1 = int(det['x1'])

            mid_x = int((x0 + x1)/2)
            mid_y = int((y0 + y1)/2)

            dist = np.square(mid_x - x) + np.square(mid_y - y)
            if (min_dist is None) or (dist < min_dist):
                min_dist = dist
                min_det = det

        if min_det is not None:
            min_det['assoc_id'] = self.fruitlet_num

        event.widget.quit()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs_path', default='../preprocess_data/pairs.json')
    parser.add_argument('--detections_path', default='../preprocess_data/pair_detections.json')
    parser.add_argument('--image_dir', default='../preprocess_data/pair_images')
    parser.add_argument('--cluster_dir', default='../preprocess_data/pair_clusters')
    parser.add_argument('--output_dir', default='../preprocess_data/pair_annotations')
    parser.add_argument('--save_default', action='store_true')
    parser.add_argument('--shuffle', action='store_false')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    pairs_path = args.pairs_path
    detections_path = args.detections_path
    image_dir = args.image_dir
    cluster_dir = args.cluster_dir
    output_dir = args.output_dir
    save_default = args.save_default
    shuffle = args.shuffle

    if not os.path.exists(pairs_path):
        raise RuntimeError('Invalid pairs_path')

    if not os.path.exists(detections_path):
        raise RuntimeError('Invalid detections_path')
    
    if not os.path.exists(image_dir):
        raise RuntimeError('Invalid image_dir')

    if not os.path.exists(cluster_dir):
        raise RuntimeError('Invalid cluster_dir')

    if not os.path.exists(output_dir):
        raise RuntimeError('Invalid output dir')

    annotate = Annotate(pairs_path, detections_path, image_dir, cluster_dir, 
                        output_dir, save_default, shuffle)
    annotate.annotate()
