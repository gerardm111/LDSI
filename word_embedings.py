import fasttext

#model = fasttext.train_unsupervised('unlabeled/tokenized_sentences.txt', model='skipgram', dim=100, epoch=10, minCount=20)
#model.save_model("model_fasttext_2.bin")
## test on:
#model = fasttext.train_unsupervised('test_tokens.txt', model='cbow', dim=100, epoch=10)
#model.save_model("model_test.bin")

model = fasttext.load_model("model_fasttext_2.bin")

## get the number of words
print("number of words in model: ", len(model.get_words())) #12634
print("---Nearest neighbors:")
list_words = ["veteran", "v.", "argues", "ptsd", "granted", "korea", "holding", "also", "va", "claim", "benefits", "act", "<NUM2>", "disability", "disease", "finds", "u.s.c.a"]
for word in list_words:
    print(word, model.get_nearest_neighbors(word, k=6))
    print("*************************")