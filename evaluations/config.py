
data_dir = '.'
data_type = 'val2017'
val_path = f'{data_dir}/annotations/captions_{data_type}.json'
subtypes = ['results', 'evalImgs', 'eval']
alg_name = 'fakecap'
[res_file, eval_imgs_file, eval_file]= \
[f'{data_dir}/results/captions_{data_type}_{alg_name}_{subtype}.json' for subtype in subtypes]

    