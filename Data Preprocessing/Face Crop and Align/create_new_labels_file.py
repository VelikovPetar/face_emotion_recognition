import os
import pandas


def verify():
    faults = 0
    with open('../../../cleaned dataset/labels.csv', mode='r') as original_labels_fname:
        original_labels = pandas.read_csv(original_labels_fname, header=None)
        with open('../../../aligned dataset/updated_labels.csv') as updated_labels_fname:
            updated_labels = pandas.read_csv(updated_labels_fname, header=None)
            original_labels_map = {}
            for _, row in original_labels.iterrows():
                original_labels_map[row[0]] = row[1]

            for _, row in updated_labels.iterrows():
                if not (row[0] in original_labels_map.keys() and original_labels_map[row[0]] == row[1]):
                    print("Invalid label for %s: Expected: %s. Actual: %s" % (row[0], original_labels_map[row[0]], row[1]))
                    faults += 1
    print('Total faults: %d' % faults)


if __name__ == '__main__':
    labels_file = '../../../cleaned dataset/labels.csv'
    images_dir = '../../../aligned dataset'
    labels_data = pandas.read_csv(labels_file, header=None)

    output_filename = images_dir + '/updated_labels.csv'
    total = 0
    with open(output_filename, mode='w') as output:
        for i, row in labels_data.iterrows():
            image_name = row[0]
            label = row[1]
            if os.path.exists(images_dir + '/' + image_name):
                output.write(image_name + ',' + label + '\n')
                total += 1

    print('Total images: %d' % total)
    verify()