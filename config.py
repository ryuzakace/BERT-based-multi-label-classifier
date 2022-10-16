import transformers

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_LABELS=5
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "/root/docker_data/model.bin"
TRAINING_FILE = "/data/jigsaw-toxic-comment-classification-challenge/train.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)