from preprocessing.preprocessing import Preprocessing
from training.gan_train import GenerativeAdversarialNetworkTrainer

autoencoder = False

preprocessing = Preprocessing(macrocategories=False, autoencoder=autoencoder, not_compatible_items_metric='random')
preprocessing.preprocess()

accessories_gan_trainer = GenerativeAdversarialNetworkTrainer(train_set_path=
                                                              '../preprocessing/json/filtered/gan_train_set_ta.json',
                                                              validation_set_path=
                                                              '../preprocessing/json/filtered/gan_validation_set_ta'
                                                              '.json',
                                                              test_set_path=
                                                              '../preprocessing/json/filtered/gan_test_set_ta.json',
                                                              autoencoder=autoencoder)
bottoms_gan_trainer = GenerativeAdversarialNetworkTrainer(train_set_path=
                                                          '../preprocessing/json/filtered/gan_train_set_tb.json',
                                                          validation_set_path=
                                                          '../preprocessing/json/filtered/gan_validation_set_tb.json',
                                                          test_set_path=
                                                          '../preprocessing/json/filtered/gan_test_set_tb.json',
                                                          autoencoder=autoencoder)
shoes_gan_trainer = GenerativeAdversarialNetworkTrainer(train_set_path=
                                                        '../preprocessing/json/filtered/gan_train_set_ts.json',
                                                        validation_set_path=
                                                        '../preprocessing/json/filtered/gan_validation_set_ts.json',
                                                        test_set_path=
                                                        '../preprocessing/json/filtered/gan_test_set_ts.json',
                                                        autoencoder=autoencoder)

accessories_gan_trainer.train('accessory')
bottoms_gan_trainer.train('bottom')
shoes_gan_trainer.train('shoes')
