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
from utils.layers_em_hinton import PrimaryCaps, ConvCaps, EMRouting, ReLUConv, ClassCaps, Squeeze
from models import original_em_capsnet_graph_smallnorb
from models.original_em_capsnet_graph_smallnorb import position_grid_conv


class TestEvalCallback(tf.keras.callbacks.Callback):
    """This callback evaluates the model at every epoch"""

    def __init__(self, test_data, output_path="test_history.json"):
        super().__init__()
        self.test_data = test_data
        self.output_path = output_path
        self.test_history = {
            'epoch': [],
            'accuracy': [],
            'loss': []
        }

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.test_data, verbose=0)
        self.test_history['epoch'].append(epoch + 1)
        self.test_history['accuracy'].append(acc)
        self.test_history['loss'].append(loss)
        print(f"\nEpoch {epoch + 1}: Test acc = {acc:.4f}, loss = {loss:.4f}")

    def on_train_end(self, logs=None):
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.test_history, f, indent=2)
        print(f"\n✅ Test history saved to: {self.output_path}")


class TestEvalCallback(tf.keras.callbacks.Callback):
    """This callback evaluates the model at every epoch"""

    def __init__(self, test_data, output_path="test_history.json"):
        super().__init__()
        self.test_data = test_data
        self.output_path = output_path
        self.test_history = {
            'epoch': [],
            'accuracy': [],
            'loss': []
        }

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.test_data, verbose=0)
        self.test_history['epoch'].append(epoch + 1)
        self.test_history['accuracy'].append(acc)
        self.test_history['loss'].append(loss)
        print(f"\nEpoch {epoch + 1}: Test acc = {acc:.4f}, loss = {loss:.4f}")

    def on_train_end(self, logs=None):
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.test_history, f, indent=2)
        print(f"\n✅ Test history saved to: {self.output_path}")
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

        model_test = EMCapsNet(model_name, mode='train', verbose=False, custom_path=custom_path)

        model_test.model.summary() 

        # ds_train, ds_test = dataset.get_tf_data()


        # We must run it like this, because the intermediate values (which are 
        # curerntly the output values) are huge. This way, tf does not store 
        # those values:)
        # for i, (x, _) in enumerate(dataset.ds_test.batch(1)):
        #     _ = model_test.predict(x)
        #     if (i+1) % 100 == 0:
        #         break

        # dataset_train, dataset_val = dataset.get_tf_data() 
        history = model_test.train(dataset=dataset, initial_epoch=0)
        
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
        # Also check the pixel next to it to see if that one is not also zeros
        self.assertTrue(np.all(tf.math.equal(expected_poses2, poses2[0, 11, 11, :])))
        self.assertFalse(np.all(tf.math.equal(expected_poses2, poses2[0, 10, 11, :])))

    def test_conv_capsule_layer(self):
        # Create a quick test model
        poses_in = tf.keras.Input(shape=(22, 22, 32, 4, 4))
        act_in = tf.keras.Input(shape=(22, 22, 32, 1, 1))
        out = ConvCaps(name="convcaps")((poses_in, act_in))
        model = tf.keras.Model(inputs=[poses_in, act_in], outputs=out)
        test_caps_in = tf.ones((1, 22, 22, 32, 4, 4)) * 5
        test_act_in = tf.ones((1, 22, 22, 32, 1, 1)) * 0.9
        print("Running model...")
        test_input = test_caps_in, test_act_in
        out = model([test_caps_in, test_act_in])
        print(out)
        print("...Ran model")
        print([t.shape for t in out])

        # print(out.shape)

    def test_gridthing(self):
         # For testing I will just run it from this
        height = 32
        width = 32
        # By adding 1 it also works for uneven numbers
        y_coords = tf.range(-height//2, (height+1)//2)
        x_coords = tf.range(-width//2, (width+1)//2)
        y, x = tf.meshgrid(x_coords, y_coords, indexing='ij')
        grid = tf.stack([x, y], axis=-1)  # shape: (H, W, 2)
        grid = tf.expand_dims(grid, axis=0)
        grid = tf.cast(grid, tf.float32)
    

        print(type(grid))
        print(grid)
        new_grid = original_em_capsnet_graph_smallnorb.position_grid(grid, 3, 1, 'VALID')
        print(new_grid)

    def test_routing_layer(self):
        # Test on small input
        # The input comes from the prev conv layer, so it is shaped
        # [batch, height, width, kernel, kernel, caps_in, caps_out, atom, atom]
        # batch is not included in the Input shape
        pose_shape = (5, 5, 3, 3, 16, 8, 4, 4)
        poses_in = tf.keras.Input(shape=pose_shape)
        # There are no activations for the out_caps yet ,so that dim is 1
        act_shape = (5, 5, 3, 3, 16, 1, 1, 1)
        act_in = tf.keras.Input(shape=act_shape)

        out = EMRouting(name="routing")((poses_in, act_in))  
        model = tf.keras.Model(inputs=[poses_in, act_in], outputs=out)

        # Fake inputs. Just random numbers, 'simulating' the start of training
        # where the values are not meaningful
        # (1)+ adds batch dimension the the shape :) (Just a batchsize 1)
        # Divide by random big number st the tensors contain small numbers

        test_pose_in = tf.random.uniform((2,)+pose_shape) / 1e4
        test_act_in = tf.random.uniform((2,)+act_shape) / 1e4

        # Set random values to zeros (I assume the model will have zeros)

        dropout_rate = 0.1
        mask = tf.cast(tf.random.uniform(tf.shape(test_pose_in)) > dropout_rate, test_pose_in.dtype)
        test_pose_in = test_pose_in * mask
        print("Running model...")
        test_input = test_pose_in, test_act_in
        out = model([test_pose_in, test_act_in])
        print(out)
        print("...Ran model")
        print([t.shape for t in out])


    def test_dataset_labels(self):
        model_name = 'SMALLNORB' 
        custom_path = None

        dataset = Dataset(model_name, config_path='config.json')
        ds_train, ds_test = dataset.get_tf_data()
        for sample in ds_train:
            print(sample)

            break

    def test_small_routing_example(self):
        """In Google Sheets, I manually created this example
        I know what the output should be, so this will check 
        if my implementation matches how I think things should work
        """
        # batch is not included in the Input shape
        pose_shape = (1, 1, 1, 1, 5, 2, 1, 2)
        poses_in = tf.keras.Input(shape=pose_shape)
        # There are no activations for the out_caps yet ,so that dim is 1
        act_shape = (1, 1, 1, 1, 5, 1, 1, 1)
        act_in = tf.keras.Input(shape=act_shape)

        out = EMRouting(name="routing")((poses_in, act_in))  
        model = tf.keras.Model(inputs=[poses_in, act_in], outputs=out)

        # Constructing the votes.
        # We only use x and y. Not rotation or anything.
        # The routing should still work
        votes = [[[[[[[[[3, 2]], [[5, 6]]], 
                      [[[5, 2]], [[1, 1]]], 
                      [[[4, 2]], [[0, 7]]], 
                      [[[8, 2]], [[3, 3]]], 
                      [[[1, 4]], [[2, 1]]]]]]]]]

        votes_tensor = tf.convert_to_tensor(votes)

        # Constructing the activation
        acts = [[[[[[ [[[0.8]]], [[[0.75]]], [[[0.9]]], [[[0.2]]], [[[0.3]]] ]]]]]]
        acts_tensor = tf.convert_to_tensor(acts)
        out = model([votes_tensor, acts_tensor])
        print(out)
        print([t.shape for t in out])

    def test_small_conv_caps(self):
        # 5x5 image, with 1 input capsules
        # Define the shape
        input_shape = (5, 5, 1, 4, 4)

        # Create the base tensor filled with zeros
        fake_input = np.zeros((1, 5, 5, 1, 4, 4), dtype=np.float32)

        # Fill the slices with increasing integers
        counter = 0
        for i in range(5):
            for j in range(5):
                fake_input[0, i, j, 0, :, :] = counter
                counter += 1

        # Convert to Tensor
        fake_input_tensor = tf.convert_to_tensor(fake_input)

        # Optionally wrap it in a Keras Input for a model
        poses = tf.keras.Input(shape=(5, 5, 1, 4, 4))
        poses_in = tf.keras.Input(shape=(5, 5, 1, 4, 4))
        act_in = tf.keras.Input(shape=(5, 5, 1, 1, 1))
        out = ConvCaps(C=1, name="convcaps")((poses_in, act_in))
        model = tf.keras.Model(inputs=[poses_in, act_in], outputs=out)

        test_act_in = tf.ones((1, 5, 5, 1, 1, 1)) * 0.5  # doesnt matter, not used

        votes, acts = model([fake_input_tensor, test_act_in])

        print(votes[0][0, 0], votes[0][0, 1])

    def test_custom_train(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

        model_name = 'SMALLNORB' 
        custom_path = None

        dataset = Dataset(model_name, config_path='config.json')

        model_test = EMCapsNet(model_name, mode='train', verbose=False, custom_path=custom_path)

        model_test.model.summary() 
 
        history = model_test.custom_train(dataset=dataset, initial_epoch=0)

    def test_simple_cnn(self):
        input_shape = [48,48,2]
        inputs = tf.keras.Input(input_shape)
        x = tf.keras.layers.Conv2D(32, 5, strides=1, activation="relu", padding='valid')(inputs)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(64, 5, strides=1, activation="relu", padding='valid')(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(5, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs,outputs=x, name='baseline_CNN')

        dataset = Dataset('SMALLNORB', config_path='config.json')
        train_ds, _ = dataset.get_tf_data()

        for images, labels in train_ds.take(1):
            print("Image shape:", images.shape)
            print("Label shape:", labels.shape)
            print("First label:", labels[0])

        images, labels = next(iter(train_ds))
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("First label:", labels[0])

        print("First image min/max:", tf.reduce_min(images[0]).numpy(), tf.reduce_max(images[0]).numpy())


        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])

        model.fit(train_ds, epochs=150)
        
    def test_simple_em_caps(self):
        """The fit() is simple, the model is the same. We also use only a small 
        amount of batches to test if we can overfit"""
        # Create position grid
        tf.random.set_seed(1004)
        input_shape = [48, 48, 2]
        height, width = input_shape[0], input_shape[1]
        x = np.linspace(-1, 1, height)
        y = np.linspace(-1, 1, width)

        position_grid = np.meshgrid(x, y)


        inputs = tf.keras.Input(input_shape)
        relu_conv1 = ReLUConv(A=64)(inputs)
        position_grid = position_grid_conv(position_grid, 5, 2, 'VALID') 
        prim_caps1 = PrimaryCaps(B=8)(relu_conv1)
        position_grid = position_grid_conv(position_grid, 1, 1, 'SAME')
        conv_caps1 = ConvCaps(C=16, stride=2)(prim_caps1)
        position_grid = position_grid_conv(position_grid, 3, 2, 'VALID')
        routing1 = EMRouting()(conv_caps1) 
    
        conv_caps2 = ConvCaps(C=16)(routing1)
        position_grid = position_grid_conv(position_grid, 3, 1, 'VALID')
        routing2 = EMRouting()(conv_caps2) 
        
        class_caps = ConvCaps(C=5, kernel_size=8)(routing2)
        # class_caps = ClassCaps(position_grid)(routing2)
        outputs = EMRouting()(class_caps) 

        outputs = Squeeze()(outputs)

        poses, acts = outputs

        # acts = tf.keras.layers.Softmax(name='softmax_output')(acts)


        # poses, acts = prim_caps1
        # reshapep = tf.keras.layers.Reshape((22, 22, 8, 16))(poses)
        # reshapea = tf.keras.layers.Reshape((22, 22, 8, 1))(acts)
        # concat = tf.keras.layers.concatenate((reshapep, reshapea))
        # flat = tf.keras.layers.Flatten()(relu_conv1)


        # acts = tf.keras.layers.Dense(5, activation='relu')(flat)
        # acts = tf.keras.layers.Softmax()(acts)

        model = tf.keras.Model(inputs=inputs,outputs=acts, name='small_EM_CapsNet')

        for var in model.trainable_variables:
            print(var.name, var.shape, var.trainable)

        use_real_data = True
        if use_real_data:
            dataset = Dataset('SMALLNORB', config_path='config.json')
            train_ds, _ = dataset.get_tf_data()

            train_ds = train_ds.take(1)
        else:
            # Create fake data: shape (1, 48, 48, 2)
            fake_image = tf.random.normal(shape=(1, 48, 48, 2))

            # Create fake one-hot label: shape (1, 5)
            fake_label = tf.one_hot(indices=[0], depth=5)  # You can replace 0 with any int 0–4

            # Create a tf.data.Dataset from the fake data
            train_ds = tf.data.Dataset.from_tensor_slices((fake_image, fake_label)).batch(1)

        for images, labels in train_ds.take(1):
            print("Image shape:", images.shape)
            print("Label shape:", labels.shape)
            print("First label:", labels[0])

            pred = model(images, training=False)
            print("Prediction:", pred.numpy())
            print("Sum over classes:", tf.reduce_sum(pred).numpy())
            print("Label:", labels.numpy())

        images, labels = next(iter(train_ds))
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("First label:", labels[0])  # 0.40460

        print("First image min/max:", tf.reduce_min(images[0]).numpy(), tf.reduce_max(images[0]).numpy())


        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        #               loss=tf.keras.losses.CategoricalCrossentropy(),
        #               metrics=['accuracy'])

        # model.fit(train_ds, epochs=150)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=5.0)  # crazy high LR for debugging

        for epoch in range(100):
            for images, labels in train_ds.take(1):  # just one batch
                with tf.GradientTape() as tape:
                    preds = model(images/10, training=True)
                    loss = loss_fn(labels, preds)
                    print(f"The loss is {loss}")

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}, Pred: {preds.numpy()}, Label: {labels.numpy()}")

                # Optional: check if gradients are zero or tiny for all weights
                for v, g in zip(model.trainable_variables, grads):
                    if g is None:
                        print(f"Warning: Gradient is None for {v.name}")
                    else:
                        print(f"{v.name}: grad mean abs {tf.reduce_mean(tf.abs(g)).numpy():.6f}")


        for images, labels in train_ds.take(1):
            print(images)
            print("Image shape:", images.shape)
            print("Label shape:", labels.shape)
            print("First label:", labels[0])

            pred = model(images, training=False)
            print("Prediction:", pred.numpy())
            print("Sum over classes:", tf.reduce_sum(pred).numpy())
            print("Label:", labels.numpy())

    def test_manual_dataset(self):
        """The Dataset object apears to be bugged: its samples are 48x48, 
        but are supposed to be 32x32. We manuall craft the dataset here to
        see if that improves model performance"""
        # Load the dataset
        (ds_train, ds_test), ds_info = tfds.load(
                'smallnorb',
                split=['train', 'test'],
                shuffle_files=True,
                as_supervised=False,
                with_info=True)
        
        b_size = 64
        # These hard coded numbers will bite me in the ass if I do not improve
        # (because if I use smaller dataset then the number of samples is wrong)
        ds_val = ds_train.take(2500)
        ds_train = ds_train.skip(2500)
        
        def preprocess_training_data(sample):
            # First we downsample to 48x48 image:
            im1 = sample['image']
            im2 = sample['image2']
            
            # Concatenate both images for the model
            images = tf.concat((im1, im2), -1, name="combine_images")

            # Downsample to 48x48
            images = tf.image.resize(images, [48, 48])

            # Normalize image to have zero mean and unit variance
            images = tf.image.per_image_standardization(images)

            # Images are 48x48, we want 32x32 patch
            # We sample the upper left corner of our patch in 0-16
            size = 32  # Size of the patch
            # specify dtype because we need ints for slicing
            corner = tf.random.uniform(shape=(2,), minval=0, maxval=48-size, dtype=tf.int32)
            x, y = corner[0], corner[1]
            images = images[y:y+size, x:x+size, :]

            # Add random brightness (0.2 is not much)
            images = tf.image.random_brightness(images, 0.2)

            images = tf.image.random_contrast(images, 0.0, 0.5)

            y = tf.one_hot(sample['label_category'], 5)
            
            return images, y
        
        def preprocess_test_data(sample):
            # First we downsample to 48x48 image:
            im1 = sample['image']
            im2 = sample['image2']
            
            # Concatenate both images for the model
            images = tf.concat((im1, im2), -1, name="combine_images")

            # Downsample to 48x48
            images = tf.image.resize(images, [48, 48])

            # Normalize image to have zero mean and unit variance
            images = tf.image.per_image_standardization(images)

            # Images are 48x48, we want 32x32 patch
            # We sample the center (indices 8-40)
            images = images[8:40, 8:40, :]

            y = tf.one_hot(sample['label_category'], 5)
            
            return images, y


        
        ds_train = ds_train.map(preprocess_training_data).batch(b_size)
        ds_test = ds_test.map(preprocess_test_data).batch(b_size)
        # Taken from training, but pre-processed like test data
        # So that it is more similar to the test data (hopefully)
        ds_val = ds_val.map(preprocess_test_data).batch(b_size)

        # DATASET is now as described by hinton. Let's see if we can train it:

        input_shape = [32, 32, 2]
        height, width = input_shape[0], input_shape[1]
        x = np.linspace(-1, 1, height)
        y = np.linspace(-1, 1, width)

        position_grid = np.meshgrid(x, y)

        inputs = tf.keras.Input(input_shape)
        relu_conv1 = ReLUConv(A=64)(inputs)
        position_grid = position_grid_conv(position_grid, 5, 2, 'VALID') 
        prim_caps1 = PrimaryCaps(B=8)(relu_conv1)
        position_grid = position_grid_conv(position_grid, 1, 1, 'SAME')
        conv_caps1 = ConvCaps(C=16, stride=2)(prim_caps1)
        position_grid = position_grid_conv(position_grid, 3, 2, 'VALID')
        routing1 = EMRouting()(conv_caps1)  # mean_data=2.722
    
        conv_caps2 = ConvCaps(C=16)(routing1)
        position_grid = position_grid_conv(position_grid, 3, 1, 'VALID')
        routing2 = EMRouting()(conv_caps2)  # mean_data=2.25

        # class_caps = ConvCaps(C=5, kernel_size=4)(routing2)
        class_caps = ClassCaps(position_grid)(routing2)
        outputs = EMRouting()(class_caps)  # mean_data=51.2

        outputs = Squeeze()(outputs)

        poses, acts = outputs

        # acts = tf.keras.layers.Softmax(name='softmax_output')(acts)


        # poses, acts = prim_caps1
        # reshapep = tf.keras.layers.Reshape((22, 22, 8, 16))(poses)
        # reshapea = tf.keras.layers.Reshape((22, 22, 8, 1))(acts)
        # concat = tf.keras.layers.concatenate((reshapep, reshapea))
        # flat = tf.keras.layers.Flatten()(relu_conv1)


        # acts = tf.keras.layers.Dense(5, activation='relu')(flat)
        # acts = tf.keras.layers.Softmax()(acts)

        model = tf.keras.Model(inputs=inputs,outputs=acts, name='small_EM_CapsNet')

        # loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()


        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=3e-3,
            decay_steps=20000,
            decay_rate=0.96
        )
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule) 
        test_callback = TestEvalCallback(ds_test)
        model.compile(loss=loss_fn, optimizer=optimizer, run_eagerly=False, metrics=['categorical_accuracy'])
        history = model.fit(ds_train, validation_data=ds_val, epochs=5, callbacks=[test_callback])
        with open("training_history.json", "w") as f:
            json.dump(history.history, f)
        # Using epsilon = 1e-7 is steady but slow, trying to anneal now. Good night! Or good morning when you read this I guess:)
        for x, y in ds_train:
            print(f"LABEL = {y}")
            pred = model.predict(x)
            print(f"PRED  = {pred}")
            break
    
        model.save('modelv_testsave.keras')
    
    def test_loading_model(self):
        model = tf.keras.models.load_model('modelv_testsave.keras')

        (ds_train, ds_test), ds_info = tfds.load(
                'smallnorb',
                split=['train', 'test'],
                shuffle_files=True,
                as_supervised=False,
                with_info=True)
        
        b_size = 64
        # These hard coded numbers will bite me in the ass if I do not improve
        # (because if I use smaller dataset then the number of samples is wrong)
        ds_val = ds_train.take(2500)
        ds_train = ds_train.skip(2500)
        
        def preprocess_training_data(sample):
            # First we downsample to 48x48 image:
            im1 = sample['image']
            im2 = sample['image2']
            
            # Concatenate both images for the model
            images = tf.concat((im1, im2), -1, name="combine_images")

            # Downsample to 48x48
            images = tf.image.resize(images, [48, 48])

            # Normalize image to have zero mean and unit variance
            images = tf.image.per_image_standardization(images)

            # Images are 48x48, we want 32x32 patch
            # We sample the upper left corner of our patch in 0-16
            size = 32  # Size of the patch
            # specify dtype because we need ints for slicing
            corner = tf.random.uniform(shape=(2,), minval=0, maxval=48-size, dtype=tf.int32)
            x, y = corner[0], corner[1]
            images = images[y:y+size, x:x+size, :]

            # Add random brightness (0.2 is not much)
            images = tf.image.random_brightness(images, 0.2)

            images = tf.image.random_contrast(images, 0.0, 0.5)

            y = tf.one_hot(sample['label_category'], 5)
            
            return images, y
        
        def preprocess_test_data(sample):
            # First we downsample to 48x48 image:
            im1 = sample['image']
            im2 = sample['image2']
            
            # Concatenate both images for the model
            images = tf.concat((im1, im2), -1, name="combine_images")

            # Downsample to 48x48
            images = tf.image.resize(images, [48, 48])

            # Normalize image to have zero mean and unit variance
            images = tf.image.per_image_standardization(images)

            # Images are 48x48, we want 32x32 patch
            # We sample the center (indices 8-40)
            images = images[8:40, 8:40, :]

            y = tf.one_hot(sample['label_category'], 5)
            
            return images, y


        
        ds_train = ds_train.map(preprocess_training_data).batch(b_size)
        ds_test = ds_test.map(preprocess_test_data).batch(b_size)
        # Taken from training, but pre-processed like test data
        # So that it is more similar to the test data (hopefully)
        ds_val = ds_val.map(preprocess_test_data).batch(b_size)

        for x, y in ds_train:
            print(f"LABEL = {y}")
            pred = model.predict(x)
            print(f"PRED  = {pred}")
            break


if __name__ == '__main__':
    suite = unittest.TestSuite()
    # suite.addTest(TestOriginalMatrixCapsules('test_primary_capsule_layer'))
    suite.addTest(TestOriginalMatrixCapsules('test_manual_dataset'))
    unittest.TextTestRunner(verbosity=2).run(suite)

