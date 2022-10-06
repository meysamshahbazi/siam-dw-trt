import numpy as np
import glob
import cv2
from easydict import EasyDict # as edict


def main():
    info = EasyDict()
    info.arch = "SiamFCRes22W"
    info.epoch_test = True
    info.cls_type = 'thinner'

    net = models.__dict__[info.arch]()
    net = load_pretrain(net, "SiamFCRes22W.pth")
    net.eval()
    net = net.cuda()
    tracker = SiamFC(info)


    path_gt = "/media/meysam/hdd/dataset/Dataset_UAV123/UAV123/anno/UAV123/car1_s.txt" 
    image_files = glob.glob("/media/meysam/hdd/dataset/Dataset_UAV123/UAV123/data_seq/UAV123/car1_s/*")
    image_files.sort()
    my_file = open(path_gt)
    line = my_file.readline()
    line = [int(l) for l in line[:-1].split(',')]
    my_file.close()

    start_frame = 0
    display_name = "image"

    for f, image_file in enumerate(image_files):
        frame = cv2.imread(image_file)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tic = cv2.getTickCount()
        if f == start_frame:  
            lx = line[0]
            ly = line[1]
            w = line[2]
            h = line[3]
            target_pos = np.array([lx + w/2, ly + h/2])
            target_sz = np.array([w, h])
            state = tracker.init(frame, target_pos, target_sz, model)  # init tracker
        else:
            frame_disp = frame.copy()

        # Draw box
            state = tracker.track(state, frame_disp)  # track
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            x1, y1, x2, y2 = int(location[0]), int(location[1]), int(location[0] + location[2]), int(location[1] + location[3])
            toc = cv2.getTickCount() - tic
            toc /= cv2.getTickFrequency()

            cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv2.putText(frame_disp, 'Tracking!', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    font_color, 1)
            cv2.putText(frame_disp, 'Press r to reset', (20, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    font_color, 1)
            cv2.putText(frame_disp, 'Press q to quit', (20, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    font_color, 1)

            # Display the resulting frame
            cv2.imshow(display_name, frame_disp)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            print('Speed: {:3.1f}fps'.format(1 / toc))
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

