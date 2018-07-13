from models import gan, gan_cls, wgan_cls, wgan, began, acgan, segan

class gan_factory(object):

    @staticmethod
    def generator_factory(type, dataset, b_size, h, scale_size, num_channels):
        if type == 'gan':
            return gan_cls.generator()
        elif type == 'wgan':
            return wgan_cls.generator(dataset=dataset)
        elif type == 'vanilla_gan':
            return gan.generator()
        elif type == 'vanilla_wgan':
            return wgan.generator()
        elif type == 'began':
            return began.Decoder(b_size, h, scale_size, num_channels)
        elif type == 'acgan':
            return segan.generator()
            #return acgan.generator()
        elif type == 'segan':
            return segan.generator()

    @staticmethod
    def discriminator_factory(type, b_size, h, scale_size, num_channels):
        if type == 'gan':
            return gan_cls.discriminator()
        elif type == 'wgan':
            return wgan_cls.discriminator()
        elif type == 'vanilla_gan':
            return gan.discriminator()
        elif type == 'vanilla_wgan':
            return wgan.discriminator()
        elif type == 'began':
            return began.Discriminator(b_size, h, scale_size, num_channels)
        elif type == 'acgan':
            return acgan.discriminator()
        elif type == 'segan':
            return gan_cls.discriminator()
