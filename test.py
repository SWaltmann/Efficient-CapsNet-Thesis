# This file serves to test all new and adapted functions. This will allow for 
# easy testing. Not all testing has been automated - some have to be checked 
# visually. I just put them here so that I can easily see the effcet of changes
# If you are not me, you probably will not care about this file at all:)

import unittest
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


from utils import pre_process_smallnorb as prep_norb
from utils import Dataset

class TestSmallNorbPreProcessing(unittest.TestCase):

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



        


if __name__ == '__main__':
    unittest.main()