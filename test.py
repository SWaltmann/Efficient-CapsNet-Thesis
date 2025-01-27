# This file serves to test all new and adapted functions. This will allow for 
# easy testing. Not all testing has been automated - some have to be checked 
# visually. I just put them here so that I can easily see the effcet of changes
# If you are not me, you probably will not care about this file at all:)

import unittest
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


from utils import pre_process_smallnorb as psn

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
            image2_list.append(image2_batch.numpy())  # Convert to NumPy and append

        # Concatenate all batches into a single NumPy array
        image2_array = np.concatenate(image2_list, axis=0)
        true_mean = image2_array.mean()
        true_std = image2_array.std()


        mean, std = psn.calculate_mean_and_std(test_dataset, "image2")

        self.assertAlmostEqual(true_mean, mean)
        self.assertAlmostEqual(true_std, std)



if __name__ == '__main__':
    unittest.main()