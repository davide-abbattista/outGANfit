from preprocessing.preprocessing import Preprocesser
from training.gan_train import GenerativeAdversarialNetworkTrainer

autoencoder = False

preprocessing = Preprocesser(macrocategories=False, autoencoder=autoencoder, not_compatible_items_metric='FID')
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

accessories_generator, ag_train_fids, ag_validation_fids, ag_test_fid = \
    accessories_gan_trainer.train_and_test('accessory')
bottoms_generator, bg_train_fids, bg_validation_fids, bg_test_fid = bottoms_gan_trainer.train_and_test('bottom')
shoes_generator, sg_train_fids, sg_validation_fids, sg_test_fid = shoes_gan_trainer.train_and_test('shoes')
