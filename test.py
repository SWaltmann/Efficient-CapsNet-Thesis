# This file serves to test all new and adapted functions. This will allow for 
# easy testing. Not all testing has been automated - some have to be checked 
# visually. I just put them here so that I can easily see the effcet of changes
# If you are not me, you probably will not care about this file at all:)

import unittest
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import json
from skimage.metrics import structural_similarity as ssim

from utils import pre_process_smallnorb as prep_norb
from utils import Dataset, plotImages, plotWrongImages
from models import EMCapsNet
from utils.layers_em_hinton import PrimaryCaps, ConvCaps, EMRouting

class TestSmallNorbPreProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n--- Starting tests in {cls.__name__} ---\n")


    def test_mean_and_std(self):
        """Test mean_and_std function against numpy.mean and numpy.std
        
        Those numpy functions cannot be used on the whole dataset due to memory 
        constraints
        """
        # Use actual dataset for accurate testing
        test_dataset = tfds.load(
                'smallnorb',
                split='train[:0.1%]',  # load very small part of the dataset
                shuffle_files=True,
                as_supervised=False,
                with_info=False)
                
        image2_list = []

        # Iterate through the dataset
        for batch in test_dataset:
            # Extract the 'image2' feature from the batch
            image2_batch = batch['image2']
            
            # Append the batch to the list
            image2_list.append(image2_batch.numpy())

        # Concatenate all batches into a single NumPy array
        image2_array = np.concatenate(image2_list, axis=0)
        true_mean = image2_array.mean()
        true_std = image2_array.std()


        mean, std = prep_norb.calculate_mean_and_std(test_dataset, "image2")
        self.assertAlmostEqual(true_mean, mean)
        self.assertAlmostEqual(true_std, std)

    def test_input_pipeline(self):
        """Ensure that the old and new standardize function do the same
        
        Walks through input pipeline, and tests after each step"""
        # Use actual dataset for accurate testing
        dataset_orig = tfds.load(
                'smallnorb',
                split='train[:0.1%]',  
                shuffle_files=True,
                as_supervised=False,
                with_info=False)
        # Work with two seperate dataset in case there are in-place operations
        dataset_stream = tfds.load(
                'smallnorb',
                split='train[:0.1%]',
                shuffle_files=True,
                as_supervised=False,
                with_info=False)
        
        
        # *_orig signifies the original methods, *_stream is the new pipeline
        # This step is skipped in the streaming pipeline - so no testing
        X_orig, y_orig = prep_norb.pre_process(dataset_orig)

        def one_hot_smallnorb(sample):
            """One hot encode the labels"""
            sample["label_category"] = tf.one_hot(sample["label_category"], 5)
            return sample
        
        dataset_stream.map(one_hot_smallnorb)

        ### Test standardize step ###

        X_orig, y_orig = prep_norb.standardize(X_orig, y_orig)

        mean_std1 = prep_norb.calculate_mean_and_std(dataset_stream, "image")
        mean_std2 = prep_norb.calculate_mean_and_std(dataset_stream, "image2")

        def standardize_smallnorb_sample(sample):
            """Basically a lambda function to pass the mean_std args
            
            This function is a wrapper around `prep_norb.standardize_sample` to 
            allow usage with `Dataset.map()`, which does not support passing 
            additional arguments. 
            """
            return prep_norb.standardize_sample(sample, mean_std1, mean_std2)

        dataset_stream = dataset_stream.map(standardize_smallnorb_sample)
        # After standardizing, the mean should be 0 and std 1:
        # I trust the mean_std fuction work because I tested it above:)
        for image in ["image", "image2"]:
            mean, std = prep_norb.calculate_mean_and_std(dataset_stream, image)
            # Checking to 2 decimal places since exact precision isnt important
            # The original data was uint8, so it had limited precision anyway
            self.assertAlmostEqual(mean, 0, places=2)
            self.assertAlmostEqual(std, 1, places=2)

        ### Test rescale step ###

        # TODO: make the path an env variable or something... 
        # or add a check for correct working dir
        with open("config.json") as json_data_file:
            config = json.load(json_data_file)

        X_orig, y_orig = prep_norb.rescale(X_orig, y_orig, config)

        def rescale_smallnorb_sample(sample):
            return prep_norb.rescale_sample(sample, config)
        
        dataset_stream = dataset_stream.map(rescale_smallnorb_sample)

        def extract_images(sample):
            image1 = sample['image']
            image2 = sample['image2']
            return image1, image2   

        extracted_dataset = dataset_stream.map(extract_images) 

        images_list = []

        for image1, image2 in extracted_dataset:
            images_list.append((image1.numpy(), image2.numpy())) 

        images_np = np.array(images_list)   

        for original, streaming in zip(X_orig, images_np):
            im1_orig = original[...,0].numpy().flatten()
            im2_orig = original[...,1].numpy().flatten()
            im1_stream = streaming[0].flatten()
            im2_stream = streaming[1].flatten()
            ssim1 = ssim(im1_orig, im1_stream, 
                         data_range=1) 
            ssim2 = ssim(im2_orig, im2_stream, 
                         data_range=1) 
            
            # ssim is a measure of how much alike two images are. They will not
            # be identical due to the different standardization.
            self.assertTrue(ssim1>0.999)
            self.assertTrue(ssim2>0.999)

        # Test test patches (crops the center region of image)
        X_orig_patch, y_orig = prep_norb.test_patches(X_orig, y_orig, config)

        res = (config['scale_smallnorb'] - config['patch_smallnorb']) // 2
        def create_smallnorb_testpatch(sample):
            return prep_norb.test_patch_sample(sample, res)

        
        dataset_stream_patch = dataset_stream.map(create_smallnorb_testpatch)

        # Reapeat the ssim thingy to see if the images are still similar after
        # extracting the patches.

        extracted_dataset = dataset_stream_patch.map(extract_images) 

        images_list = []

        for image1, image2 in extracted_dataset:
            images_list.append((image1.numpy(), image2.numpy())) 

        images_np = np.array(images_list)   

        for original, streaming in zip(X_orig_patch, images_np):
            im1_orig = original[...,0].numpy().flatten()
            im2_orig = original[...,1].numpy().flatten()
            im1_stream = streaming[0].flatten()
            im2_stream = streaming[1].flatten()
            ssim1 = ssim(im1_orig, im1_stream, 
                         data_range=1) 
            ssim2 = ssim(im2_orig, im2_stream, 
                         data_range=1) 
            
            # ssim is a measure of how much alike two images are. They will not
            # be identical due to the different standardization.
            self.assertTrue(ssim1>0.999)
            self.assertTrue(ssim2>0.999)


class TestOriginalMatrixCapsules(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n--- Starting tests in {cls.__name__} ---\n")

    def test_model(self):
        # os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

        model_name = 'SMALLNORB' 
        custom_path = None

        dataset = Dataset(model_name, config_path='config.json')

        model_test = EMCapsNet(model_name, mode='test', verbose=False, custom_path=custom_path)

        model_test.model.summary() 

        # train, test = dataset.get_tf_data()  # TODO: figure out if there is any advantage to having to call this manually
        
        # The Dataset is pretty unintuitive - having to manually batch it 
        # seems... dumb

        # Also, we cannot predict using the train set, because that one includes
        # the y_true label, which this model does not use. Probably better to
        # make a new repo

        # We must run it like this, because the intermediate values (which are 
        # curerntly the output values) are huge. This way, tf does not store 
        # those values:)
        for i, (x, _) in enumerate(dataset.ds_test.batch(1)):
            _ = model_test.predict(x)
            if (i+1) % 100 == 0:
                break
        
    def test_primary_capsule_layer(self):

        # Create fake input - all ones for image1, all twos for image2
        # First dimension represents the batch size (=1 for this)
        image1 = tf.ones((1, 48, 48, 1))
        image2 = tf.ones((1, 48, 48, 1)) * 2

        test_input = tf.concat([image1, image2], axis=-1)
        # We now have a 48x48 'image', with 2 channels.

        # Create a quick test model
        x = tf.keras.Input(shape=(48, 48, 2))
        out = PrimaryCaps(name="primary_caps")(x)

        model = tf.keras.Model(inputs=x, outputs=out)
        
        # Overwrite the kernel, so that all this layer does is shuffle numbers
        # I want to make sure that numbers end up in the right place after 
        # passing through this layer
        prim_caps = model.get_layer("primary_caps")
        prim_caps.conv.kernel.assign(tf.ones((prim_caps.kernel_size,
                                              prim_caps.kernel_size,
                                              2,  # num of input channels
                                              17*32)))  # (16+1) * num output capsules
        
        # This kernel should just add the channels together for the poses,
        # So the poses matrices should be filled with 1+2 = 3s
        # Moreover, we should have 32 of those capsules at each grid position
        # We use SAME padding (which just extends the ones and twos of the input)
        # So the shape (1, 48, 48) should be the same as the input

        expected_poses = tf.ones((1, 48, 48, 32, 4, 4)) * 3

        # For the activations, the same thing goes, but they undergo a sigmoid
        # so they should all be sigmoid(3) = 0.952574126

        sig = 0.952574126

        expected_activations = tf.ones((1, 48, 48, 32, 1, 1)) * sig

        # Run it
        output = model(test_input)
        print([t.shape for t in output])
        poses, activations = output

        self.assertTrue(np.all(tf.math.equal(expected_poses, poses)))
        self.assertTrue(np.all(tf.math.equal(expected_activations, activations)))

        # Now we should also check to make sure that input values at specific 
        # locations influence the corresponding capsules. For this, we set
        # a random 'pixel' to 0 (both channels). The output at that position should
        # be all zeroes, ensuring that the reshaping all worked out correctly
        test_input2 = test_input.numpy()
        test_input2[0, 11, 11, :] = np.zeros((1, 1, 2))
        test_input2 = tf.convert_to_tensor(test_input2)

        poses2, activations2 = model(test_input2)

        # The poses and activation of the capsule at that location should be 0
        # print(output2.shape)
        expected_poses2 = tf.zeros((32, 4, 4))
        # Check the pixel next to it to see if that one is not also zeros
        self.assertTrue(np.all(tf.math.equal(expected_poses2, poses2[0, 11, 11, :])))
        self.assertFalse(np.all(tf.math.equal(expected_poses2, poses2[0, 10, 11, :])))



if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestOriginalMatrixCapsules('test_primary_capsule_layer'))
    unittest.TextTestRunner(verbosity=2).run(suite)

