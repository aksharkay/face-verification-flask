from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp


config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "edgeface_xs_gamma_06"
config.resume = False
config.output = 'edgeface_xs_gamma_06/'
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = False
config.weight_decay = 0.05
config.batch_size = 512
config.optimizer = "adamw"
config.lr = 3e-3
config.verbose = 2000
config.dali = True 
config.dali_aug = True

config.num_workers = 6

config.rec = "data/webface12m"
config.num_classes = 617970
config.num_image = 12720066
config.num_epoch = 100
config.warmup_epoch = 2
config.val_targets = []

