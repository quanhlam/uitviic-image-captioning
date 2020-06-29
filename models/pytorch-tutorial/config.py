

class Config(object):
    # Paths
    img_path = '../../Images/resized2017'
    train_path = 'data/annotations/captions_train2017.json'
    val_path = 'data/annotations/captions_val2017.json'
    test_path = ''
    vocab_path = 'data/vocab.pkl'
    model_path = 'models/'
    machine_output_path = 'data/machine_output.json'
    tokenizer = 'pyvi'

    optimizer = "adam"

    threshold = 1
    learning_rate = 0.001
    num_epochs = 15
    crop_size = 224
    batch_size = 128
    embed_size = 256 # Dimension of word embedding vectors
    hidden_size = 512 # Dimension of lstm hidden states
    num_layers = 1 # NUmber of layers in lstm   
    num_workers = 2

    save_step = 1

    log_step = 10

    encoder_path = 'models/encoder-15-4.ckpt'
    decoder_path = 'models/decoder-15-4.ckpt'

