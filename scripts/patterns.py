import os
import re


ranked_confidences_pattern = re.compile(
    r'.*(?P<split>\d)_r(?P<rank>\d)_confidences.csv')
ranked_votes_pattern = re.compile(r'.*(?P<split>\d)_r(?P<rank>\d)_votes.csv')
ranked_confusion_matrix_pattern = re.compile(
    r'.*(?P<split>\d)_r(?P<rank>\d)_confusion_matrix.csv')

confidences_pattern = r'(?P<split>\d)_confidences.csv'
votes_pattern = r'(?P<split>\d)_votes.csv'


def split_paths(csv_paths):
    confidence_csv_paths = [
        path for path in csv_paths if ranked_confidences_pattern.match(os.path.basename(path))]
    vote_csv_paths = [
        path for path in csv_paths if ranked_votes_pattern.match(os.path.basename(path))]
    confusion_matrix_paths = [
        path for path in csv_paths if ranked_confusion_matrix_pattern.match(os.path.basename(path))]

    return confidence_csv_paths, vote_csv_paths, confusion_matrix_paths
