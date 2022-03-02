import sys, argparse
from joblib import load
import fasttext
import luima_sbd.sbd_utils as luima
from utils.preprocessing import one_sentence_tokenizer
from utils.word_embedding_featurization import make_feature_vectors
import spacy
from spacy.attrs import ORTH
import pandas as pd

#path_txt_file = sys.argv[1]
parser=argparse.ArgumentParser()
parser.add_argument('path', metavar='N', type=str, nargs='+', help='The path to the BVA decision to test')
parser.add_argument('-csv','--write_to_csv', default=False, type=bool, help='Where you want to have the output. If True the output is saved in a csv file instead of writting it in the terminal')
parser.add_argument('-m', '--model', default='LR', type=str, help="The model you want to use for prediction. Either 'LR' for Logistic Regression (default) or 'RF' for Random Forest")
args=parser.parse_args()
path_txt_file = args.path[0]

with open(path_txt_file, 'r') as file:
    BVA_txt = file.read()

# sentence segmentation
list_sentences = luima.text2sentences(BVA_txt)
list_sentences_idx = luima.text2sentences(BVA_txt, offsets=True)

# nlp
nlp = spacy.load("en_core_web_sm")
nlp.tokenizer.add_special_case('Vet. App.', [{ORTH: 'Vet. App.'}])
nlp.tokenizer.add_special_case('Veterans Law Judge', [{ORTH: 'Veterans Law Judge'}])
nlp.tokenizer.add_special_case('Veterans Affairs', [{ORTH: 'Veterans Affairs'}])
nlp.tokenizer.add_special_case("Veterans' Appeals", [{ORTH: "Veterans' Appeals"}])
ruler = nlp.get_pipe("attribute_ruler")
patterns = [[{"TEXT": "["}], [{"TEXT": "\n"}], [{"TEXT": "'"}], [{"TEXT": "\r"}], [{"TEXT": "\t"}]]
attrs = {"POS": "PUNCT"}
ruler.add(patterns=patterns, attrs=attrs, index=0)
nlp.disable_pipes('parser')
# spans creation
doc_lenght = len(BVA_txt)
document_list_of_dict = []
for i in range(len(list_sentences)):
    sentence_dict = {}
    sentence_dict['txt'] = list_sentences[i]
    sentence_dict['start_normalized']= list_sentences_idx[i][0]/doc_lenght
    sentence_dict['tokens_spacy'] = one_sentence_tokenizer(list_sentences[i], nlp)
    document_list_of_dict.append(sentence_dict)

# word embeddings
fasttext_model = fasttext.load_model("./models/model_fasttext.bin")
test_X = make_feature_vectors(document_list_of_dict, fasttext_model)
print("shape of X: ", f'{test_X.shape}')

# prediction
if args.model == 'RF':
    classification_model = load('./models/random_forest_best.joblib')
    print('---Random Forest Classifier used')
else:
    classification_model = load('./models/logistic_regression_best.joblib')
    print('---Logistic Regression Classifier used')
prediction_list = classification_model.predict(test_X)

# print predicted class
display_list = [list_sentences, prediction_list]
df = pd.DataFrame (display_list).transpose()
df.columns = ['sentence', 'predicted type']
if args.write_to_csv:
    df.to_csv('./outputs/output_predicted_type.csv')
    print("Results printed in csv file 'output_predicited_type.csv' in folder 'outputs'")
else:
    print(df)