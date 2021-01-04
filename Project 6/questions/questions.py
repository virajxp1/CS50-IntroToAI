from operator import itemgetter
import nltk
import os
import string
import numpy as np
import sys
from nltk.corpus import stopwords

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():
    # Check command-line arguments

    dir = "corpus"

    # Calculate IDF values across files
    files = load_files(dir)
    file_words = {
        filename: tokenize_start(files[filename],filename)
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    while True:
    # Prompt user for query
        query = set(tokenize(input("Query: ")))

        if len(query) == 0:
            break

        # Determine top file matches according to TF-IDF
        filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

        # Extract sentences from top files
        sentences = dict()
        for filename in filenames:
            for passage in files[filename].split("\n"):
                for sentence in nltk.sent_tokenize(passage):
                    tokens = tokenize(sentence)
                    if tokens:
                        sentences[sentence] = tokens

        # Compute IDF values across sentences
        idfs = compute_idfs(sentences)

        # Determine top sentence matches
        matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
        for match in matches:
            print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_to_filetxt = {}
    cwd = os.getcwd()
    path = os.path.join(cwd, directory)
    for file_name in os.listdir(path):
        with open(os.path.join(path, file_name), encoding="utf8") as f:
            file_text = f.readlines()
        file_text = ' '.join([str(elem) for elem in file_text])
        file_to_filetxt[file_name] = file_text
    return file_to_filetxt

def tokenize_start(document,filename):
    print("Loading files from {}".format(filename))
    return tokenize(document)

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    document = document.lower()
    tokens = nltk.word_tokenize(document)
    tokens = [word for word in tokens if
              not word in nltk.corpus.stopwords.words('english') and word not in string.punctuation and len(word) > 1]
    return tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    # dictionary to hold words and the number of documents they appear in

    doc_freq = {}
    total_documents = 0
    for doc in documents:
        total_documents += 1
        doc_words = documents[doc]
        seen_words = set()
        for word in doc_words:
            if word not in seen_words:
                seen_words.add(word)
                if word not in doc_freq:
                    doc_freq[word] = 1
                else:
                    doc_freq[word] += 1
    for key in doc_freq:
        x = doc_freq[key]
        x = np.log(total_documents / x)
        doc_freq[key] = x

    return doc_freq


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    # for each document file in files
    # take each word in the query
    # if the word also appears in the document calculate the tf-idf for that word
    # score each file by the sum of all tf-idf scores of each word in both the query and that document

    doc_scores = {}

    for document_name in files:
        Words_in_file = files[document_name]
        doc_scores[document_name] = 0
        for word in query:
            # check does the word appear in the document?
            if word in Words_in_file:
                # calculate tf-idf score
                # tf = term frequency need to calculate this
                # tf = number of times that word appears in file_words
                tf = Words_in_file.count(word)
                tf_idf = tf * idfs[word]
                doc_scores[document_name] += tf_idf
    res = dict(sorted(doc_scores.items(), key=itemgetter(1), reverse=True)[:n])
    topRanked = []
    for key in res:
        topRanked.append(key)
    return topRanked


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    # here we are scoring sentences

    sentence_scores = {}

    for sentence in sentences:
        words_in_sentence = sentences[sentence]
        sentence_scores[sentence] = (0,0)
        for word in query:
            if word in words_in_sentence:
                idf_score = idfs[word]
                current_score = sentence_scores[sentence][0]
                query_score = getQueryScore(query,words_in_sentence)
                sentence_scores[sentence] = (current_score+idf_score,query_score,len(words_in_sentence))
    res = dict(sorted(sentence_scores.items(), key=itemgetter(1), reverse=True)[:n])
    topRanked = []
    for key in res:
        topRanked.append(key)
    return topRanked

def getQueryScore(query,sentence):
    wordsInQuery = 0
    for word in sentence:
        if word in query:
            wordsInQuery+=1
    return wordsInQuery/len(sentence)

if __name__ == "__main__":
    main()
