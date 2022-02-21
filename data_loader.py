
import yaml
import json

## Creating the dataset with training docs only (excluding test and dev sets)
def train_data_loader(data, path_to_exclude):
    """
    :param data: corpus of documents
    :type data: dictionary
    :param path_to_exclude: path to the yaml documents containing Ids of files to exclude
    :type path_to_exclude: str

    :output: dict of training documents (i.e. without test and dev documents)
    """
    ## creating list of IDs of train and dev sets
    with open(path_to_exclude) as stream:
        data_loaded = yaml.safe_load(stream)
    train_and_dev_doc_ids = []
    for key in data_loaded:
        for elem in data_loaded[key]:
            train_and_dev_doc_ids.append(elem)
    # print(len(train_and_dev_doc_ids))
    # print(train_and_dev_doc_ids)

    data_train={'documents': [], 'annotations': [], 'types': []}
    data_train.keys()
    for document in data['documents']:
        if not (document['_id'] in set(train_and_dev_doc_ids)):
            data_train['documents'].append(document)
    for annotation in data['annotations']:
        if not (annotation['document'] in set(train_and_dev_doc_ids)):
            data_train['annotations'].append(annotation)
    data_train['types']=data['types']
    return data_train

def save_data_as_json(data):
    with open('labeled/ldsi_w21_train_annotations.json') as file:
        json.dump(data, file, indent=2)