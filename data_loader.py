
import yaml
import json

## Creating the dataset with training docs only (excluding test and dev sets)
def train_data_loader(data, path_to_exclude, set_of_data="train"):
    """
    :param data: corpus of documents
    :type data: dictionary
    :param path_to_exclude: path to the yaml documents containing Ids of files to exclude
    :type path_to_exclude: str
    :param set_of_data: whether it is the "train", "dev" or "test" set [default: "train"]
    :type set_of_data: str

    :output: dict of training/development/testing documents (i.e. without documents from other sets)
    """
    ## creating list of IDs of train and dev sets
    with open(path_to_exclude) as stream:
        data_loaded = yaml.safe_load(stream)

    # print(len(train_and_dev_doc_ids))
    # print(train_and_dev_doc_ids)
    data_train={'documents': [], 'annotations': [], 'types': []}
    if set_of_data == "train":
        train_and_dev_doc_ids = []
        for key in data_loaded:
            for elem in data_loaded[key]:
                train_and_dev_doc_ids.append(elem)
        for document in data['documents']:
            if not (document['_id'] in set(train_and_dev_doc_ids)):
                data_train['documents'].append(document)
        for annotation in data['annotations']:
            if not (annotation['document'] in set(train_and_dev_doc_ids)):
                data_train['annotations'].append(annotation)
        data_train['types']=data['types']
        print("---Training data loaded")

    elif set_of_data == "dev":
        dev_doc_ids = []
        for elem in data_loaded[set_of_data]:
                dev_doc_ids.append(elem)
        for document in data['documents']:
            if (document['_id'] in set(dev_doc_ids)):
                data_train['documents'].append(document)
        for annotation in data['annotations']:
            if (annotation['document'] in set(dev_doc_ids)):
                data_train['annotations'].append(annotation)
        data_train['types']=data['types']
        print("---Development data loaded")

    elif set_of_data == "test":
        test_doc_ids = []
        for elem in data_loaded[set_of_data]:
                test_doc_ids.append(elem)
        for document in data['documents']:
            if (document['_id'] in set(test_doc_ids)):
                data_train['documents'].append(document)
        for annotation in data['annotations']:
            if (annotation['document'] in set(test_doc_ids)):
                data_train['annotations'].append(annotation)
        data_train['types']=data['types']
        print("---Testing data loaded")
    return data_train

def save_data_as_json(data):
    with open('labeled/ldsi_w21_train_annotations.json') as file:
        json.dump(data, file, indent=2)

# get all sentences assuming every annotation is a sentence
def make_span_data(data):
    span_data = []
    cpt = 0
    documents_by_id = {d['_id']: d for d in data['documents']}
    types_by_id = {t['_id']: t for t in data['types']}
    annotations = data['annotations']
    for a in annotations:
        print("annotation: ", cpt, " /14 291", end="\r")
        start = a['start']
        end = a['end']
        document_txt = documents_by_id[a['document']]['plainText']
        atype = a['type']
        sd = {'txt': document_txt[start:end],
              'document': a['document'],
              'type': types_by_id[atype]['name'],
              'start': a['start'],
              'start_normalized': a['start'] / len(document_txt),
              'end': a['end']}
        span_data.append(sd)
        cpt += 1
    span_labels = [s['type'] for s in span_data]
    span_txt = [s['txt'] for s in span_data]
    print("---Spanning dataset completed")
    return span_data, span_labels, span_txt