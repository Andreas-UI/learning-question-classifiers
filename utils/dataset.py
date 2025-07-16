from torch.utils.data import Dataset
from sklearn import preprocessing
import numpy as np
from utils.text import Tokenizer
from utils.sequence import pad_sequences


class LQCDataset(Dataset):
    def __init__(
        self, data, label_encoder=None, tokenizer=None, max_sentence_length=None
    ):
        self.questions = list()
        max_length = 0

        self.labels = list()
        self.numeral_labels = list()
        self.string_classes = list()

        for row in data:
            label, question = self.label_question_split(row)
            question = question.lower()

            self.labels.append(label)
            self.questions.append(question)

            if max_length < len(question):
                max_length = len(question)

        self.max_sentence_length = (
            max_length if max_sentence_length is None else max_sentence_length
        )

        # Label Encoding
        if label_encoder is None:
            label_encoder = preprocessing.LabelEncoder()
            label_encoder.fit(self.labels)

        self.numeral_labels = np.array(label_encoder.transform(self.labels))
        self.string_classes = label_encoder.classes_
        self.num_classes = len(label_encoder.classes_)

        # Tokenization
        if tokenizer is None:
            tokenizer = Tokenizer(oov_token="<UNK>")
            tokenizer.fit_on_texts(self.questions)

        self.numeral_questions = tokenizer.texts_to_sequences(self.questions)
        self.numeral_questions = pad_sequences(
            self.numeral_questions,
            padding="post",
            truncating="post",
            maxlen=self.max_sentence_length,
        )

        self.word2idx = tokenizer.word_index
        self.word2idx = {k: v for k, v in self.word2idx.items()}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        self.label_encoder = label_encoder
        self.tokenizer = tokenizer

    def label_question_split(self, string):
        separator_idx = 0
        for i in range(len(string)):
            if string[i] == ":":
                separator_idx = i
                break

        return string[:separator_idx], string[separator_idx + 1 :]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.numeral_questions[idx], self.numeral_labels[idx]
