import os
import pandas as pd


image_folderpath = "PR1-Strawberry/Images"
uncleaned_csv_path = "PR1-Strawberry/Annotations_copy.csv"


def split_name(name):
    """
    Split the filename string and get the image name, image index, image label
    """
    string = name.replace(".PNG", "")
    parts = string.split("_")
    index = parts[0] + "_" + parts[1]
    label = parts[2]
    return name, index, label


if __name__ == "__main__":
    clean_total_list = []
    for filename in os.listdir(image_folderpath):
        image_path = os.path.join(image_folderpath, filename)
        name_cleaned = os.path.basename(image_path)

        name_cleaned, index_cleaned, label_cleaned = split_name(name_cleaned)
        # get a huge list, including three parts

        clean_total_list.append(name_cleaned)
        clean_total_list.append(index_cleaned)
        clean_total_list.append(label_cleaned)

    uncleaned_csv = pd.read_csv(uncleaned_csv_path)
    # read uncleaned_csv
    name_uncleaned = uncleaned_csv.iloc[:, 0]
    name_uncleaned_list = name_uncleaned.tolist()

    unclean_total_list = []
    for item in name_uncleaned_list:
        name_uncleaned, index_ubcleaned, label_uncleaned = split_name(item)
        # get a huge list, including three parts
        unclean_total_list.append(name_uncleaned)
        unclean_total_list.append(index_ubcleaned)
        unclean_total_list.append(label_uncleaned)

    num_images = len(clean_total_list) // 3

    for i in range(num_images):
        # find the corresponding index
        # index can help us find the image where there is an error
        element_to_find = clean_total_list[3 * i + 1]
        for index, element in enumerate(unclean_total_list):
            if (
                # use index to match corresponding images
                element == element_to_find
                # inspect
                and clean_total_list[3 * i] != unclean_total_list[index - 1]
            ):
                print("clean:", clean_total_list[3 * i])
                print("unclean:", unclean_total_list[index - 1])
                # clean the noise
                unclean_total_list[index - 1] = clean_total_list[3 * i]
                unclean_total_list[index + 1] = clean_total_list[3 * i + 2]
    # print("Finally get the clean list:", unclean_total_list)

    new_filename = []
    new_label = []
    # extract two columns of data from the list and write them back to the original file
    for j in range(len(unclean_total_list)):
        if j % 3 == 0:
            new_filename.append(unclean_total_list[j])
            new_label.append(unclean_total_list[j + 2])

    uncleaned_csv["file_name"] = pd.DataFrame(new_filename, columns=["file_name"])
    uncleaned_csv["label"] = pd.DataFrame(new_label, columns=["label"])

    uncleaned_csv.to_csv("PR1-Strawberry/Annotations_copy.csv", index=False)
