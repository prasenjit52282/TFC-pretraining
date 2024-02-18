class Config(object):
    def __init__(self):
        # model configs
        self.kernel_size = 8  #-- not in TFC
        self.stride = 1  #-- not in TFC
        self.final_out_channels = 128  #-- not in TFC

        self.num_classes = 10 #-- not in TFC
        self.dropout = 0.35   #-- not in TFC
        self.features_len = 18   #-- not in TFC

        # training configs  ------TFC parameters starts here
        self.input_channels = 10# num gases sensed along with temp and hum
        self.num_epoch = 100

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128

        #Hyperparameters
        self.TSlength_aligned = 300

        self.target_batch_size = 64      #only for finetuning
        self.num_classes_target = 10     #only for finetuning

        self.Context_Cont = Context_Cont_configs()
        self.augmentation = augmentations() #----TFC parameters ends here


        #"""Other hyperparameters"""
        self.TC = TC() #-- not in TFC
        self.lr_f = self.lr  #-- not in TFC
        self.increased_dim = 1  #-- not in TFC
        self.final_out_channels = 128  #-- not in TFC
        self.features_len_f = self.features_len #-- not in TFC
        self.CNNoutput_channel = 28#  104 #-- not in TFC


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 6
