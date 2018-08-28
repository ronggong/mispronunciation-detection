import csv


def open_csv_recordings(filename):
    recordings = []
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            recordings.append(row)
    return recordings


def write_csv_two_columns_list(two_columns_list, filename):
    with open(filename, 'wb') as csvfile:
        two_columns_writer = csv.writer(csvfile, delimiter=',')
        for l in two_columns_list:
            two_columns_writer.writerow(l)
