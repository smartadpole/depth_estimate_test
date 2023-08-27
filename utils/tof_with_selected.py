import cv2

def selected_tof( tof_file, tof_selected_file):
    image = cv2.imread(tof_file)
    image_selected = cv2.imread(tof_selected_file
                                )


def main():
    image_tof = ""
    image_selected = ""
    image_save = selected_tof(image_tof, image_selected)
    cv2.imwrite("write.png", image_save)
if __name__ == '__main__':
    main()