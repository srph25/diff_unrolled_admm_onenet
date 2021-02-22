import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io
import scipy.misc
import scipy.ndimage
import layers_nearest as layers
import os
import os.path
import timeit
from multiprocessing import Process, Queue
import argparse
from smooth_stream import SmoothStream
import sys
sys.path.append('../admm')
import add_noise as add_noise_admm
import inpaint as problem_inpaint
import inpaint_center as problem_inpaint_center
import inpaint_block as problem_inpaint_block
import superres as problem_superres
from noise import add_noise
import datetime
#from scipy import sparse
from pil import fromimage, toimage, imresize, imread, imsave

def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable


def get_session():
    """Returns the TF session to be used by the backend.
    If a default TensorFlow session is available, we will return it.
    Else, we will return the global Keras session.
    If no global Keras session exists at this point:
    we will create a new global session.
    Note that you can manually set the global session
    via `K.set_session(sess)`.
    # Returns
        A TensorFlow session.
    """
    _SESSION = None
    _MANUAL_VAR_INIT = False

    default_session = tf.get_default_session()

    if default_session is not None:
        session = default_session
    else:
        if _SESSION is None:
            if not os.environ.get('OMP_NUM_THREADS'):
                config = tf.ConfigProto(allow_soft_placement=True)
            else:
                num_thread = int(os.environ.get('OMP_NUM_THREADS'))
                config = tf.ConfigProto(intra_op_parallelism_threads=num_thread,
                                        inter_op_parallelism_threads=num_thread,
                                        allow_soft_placement=True)
            _SESSION = tf.Session(config=config)
        session = _SESSION
    if not _MANUAL_VAR_INIT:
        with session.graph.as_default():
            variables = tf.global_variables()
            candidate_vars = []
            for v in variables:
                if not getattr(v, '_keras_initialized', False):
                    candidate_vars.append(v)
            if candidate_vars:
                # This step is expensive, so we only run it on variables
                # not already marked as initialized.
                is_initialized = session.run(
                    [tf.is_variable_initialized(v) for v in candidate_vars])
                uninitialized_vars = []
                for flag, v in zip(is_initialized, candidate_vars):
                    if not flag:
                        uninitialized_vars.append(v)
                    v._keras_initialized = True
                if uninitialized_vars:
                    session.run(tf.variables_initializer(uninitialized_vars))
    # hack for list_devices() function.
    # list_devices() function is not available under tensorflow r1.3.
    if not hasattr(session, 'list_devices'):
        session.list_devices = lambda: device_lib.list_local_devices()
    return session

def _get_available_devices():
    return [x.name for x in get_session().list_devices()]


def _normalize_device_name(name):
    name = '/' + ':'.join(name.lower().replace('/', '').split(':')[-2:])
    return name


def multi_gpu_model(inputs, slice_input, len_outputs, model, gpus=None, cpu_merge=True, cpu_relocation=False):
    """Replicates a model on different GPUs.
    Specifically, this function implements single-machine
    multi-GPU data parallelism. It works in the following way:
    - Divide the model's input(s) into multiple sub-batches.
    - Apply a model copy on each sub-batch. Every model copy
        is executed on a dedicated GPU.
    - Concatenate the results (on CPU) into one big batch.
    E.g. if your `batch_size` is 64 and you use `gpus=2`,
    then we will divide the input into 2 sub-batches of 32 samples,
    process each sub-batch on one GPU, then return the full
    batch of 64 processed samples.
    This induces quasi-linear speedup on up to 8 GPUs.
    This function is only available with the TensorFlow backend
    for the time being.
    # Arguments
        model: A Keras model instance. To avoid OOM errors,
            this model could have been built on CPU, for instance
            (see usage example below).
        gpus: Integer >= 2 or list of integers, number of GPUs or
            list of GPU IDs on which to create model replicas.
        cpu_merge: A boolean value to identify whether to force
            merging model weights under the scope of the CPU or not.
        cpu_relocation: A boolean value to identify whether to
            create the model's weights under the scope of the CPU.
            If the model is not defined under any preceding device
            scope, you can still rescue it by activating this option.
    # Returns
        A Keras `Model` instance which can be used just like the initial
        `model` argument, but which distributes its workload on multiple GPUs.
    # Examples
    Example 1 - Training models with weights merge on CPU
    ```python
        import tensorflow as tf
        from keras.applications import Xception
        from keras.utils import multi_gpu_model
        import numpy as np
        num_samples = 1000
        height = 224
        width = 224
        num_classes = 1000
        # Instantiate the base model (or "template" model).
        # We recommend doing this with under a CPU device scope,
        # so that the model's weights are hosted on CPU memory.
        # Otherwise they may end up hosted on a GPU, which would
        # complicate weight sharing.
        with tf.device('/cpu:0'):
            model = Xception(weights=None,
                             input_shape=(height, width, 3),
                             classes=num_classes)
        # Replicates the model on 8 GPUs.
        # This assumes that your machine has 8 available GPUs.
        parallel_model = multi_gpu_model(model, gpus=8)
        parallel_model.compile(loss='categorical_crossentropy',
                               optimizer='rmsprop')
        # Generate dummy data.
        x = np.random.random((num_samples, height, width, 3))
        y = np.random.random((num_samples, num_classes))
        # This `fit` call will be distributed on 8 GPUs.
        # Since the batch size is 256, each GPU will process 32 samples.
        parallel_model.fit(x, y, epochs=20, batch_size=256)
        # Save model via the template model (which shares the same weights):
        model.save('my_model.h5')
    ```
    Example 2 - Training models with weights merge on CPU using cpu_relocation
    ```python
         ..
         # Not needed to change the device scope for model definition:
         model = Xception(weights=None, ..)
         try:
             parallel_model = multi_gpu_model(model, cpu_relocation=True)
             print("Training using multiple GPUs..")
         except ValueError:
             parallel_model = model
             print("Training using single GPU or CPU..")
         parallel_model.compile(..)
         ..
    ```
    Example 3 - Training models with weights merge on GPU (recommended for NV-link)
    ```python
         ..
         # Not needed to change the device scope for model definition:
         model = Xception(weights=None, ..)
         try:
             parallel_model = multi_gpu_model(model, cpu_merge=False)
             print("Training using multiple GPUs..")
         except:
             parallel_model = model
             print("Training using single GPU or CPU..")
         parallel_model.compile(..)
         ..
    ```
    # On model saving
    To save the multi-gpu model, use `.save(fname)` or `.save_weights(fname)`
    with the template model (the argument you passed to `multi_gpu_model`),
    rather than the model returned by `multi_gpu_model`.
    """

    available_devices = _get_available_devices()
    available_devices = [_normalize_device_name(name)
                         for name in available_devices]
    if not gpus:
        # Using all visible GPUs when not specifying `gpus`
        # e.g. CUDA_VISIBLE_DEVICES=0,2 python keras_mgpu.py
        gpus = len([x for x in available_devices if 'gpu' in x])

    if isinstance(gpus, (list, tuple)):
        if len(gpus) <= 1:
            raise ValueError('For multi-gpu usage to be effective, '
                             'call `multi_gpu_model` with `len(gpus) >= 2`. '
                             'Received: `gpus=%s`' % gpus)
        num_gpus = len(gpus)
        target_gpu_ids = gpus
    else:
        if gpus <= 1:
            '''
            raise ValueError('For multi-gpu usage to be effective, '
                             'call `multi_gpu_model` with `gpus >= 2`. '
                             'Received: `gpus=%d`' % gpus)
            '''
            print('For multi-gpu usage to be effective, '
                             'call `multi_gpu_model` with `gpus >= 2`. '
                             'Received: `gpus=%d`' % gpus)
        num_gpus = gpus
        target_gpu_ids = range(num_gpus)

    #import tensorflow as tf

    target_devices = ['/cpu:0'] + ['/gpu:%d' % i for i in target_gpu_ids]
    for device in target_devices:
        if device not in available_devices:
            raise ValueError(
                'To call `multi_gpu_model` with `gpus=%s`, '
                'we expect the following devices to be available: %s. '
                'However this machine only has: %s. '
                'Try reducing `gpus`.' % (gpus,
                                          target_devices,
                                          available_devices))

    def get_slice(data, i, parts):
        shape = tf.shape(data)
        batch_size = shape[:1]
        input_shape = shape[1:]
        step = batch_size // parts
        if i == parts - 1:
            size = batch_size - step * i
        else:
            size = step
        size = tf.concat([size, input_shape], axis=0)
        stride = tf.concat([step, input_shape * 0], axis=0)
        start = stride * i
        return tf.slice(data, start, size)

    def to_list(x, allow_tuple=False):
        if isinstance(x, list):
            return x
        if allow_tuple and isinstance(x, tuple):
            return list(x)
        return [x]

    # Relocate the model definition under CPU device scope if needed
    '''
    if cpu_relocation:
        with tf.device('/cpu:0'):
            inputs, outputs = model = clone_model(model)
    '''

    all_outputs = []
    for i in range(len_outputs):
        all_outputs.append([])

    # Place a copy of the model on each GPU,
    # each getting a slice of the inputs.
    for i, gpu_id in enumerate(target_gpu_ids):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('replica_%d' % gpu_id):
                _inputs = []
                # Retrieve a slice of the input.
                for j, x in enumerate(inputs):
                    if slice_input[j] is True:
                        # In-place input splitting which is not only
                        # 5% ~ 12% faster but also less GPU memory
                        # duplication.
                        with tf.device(x.device):
                            input_shape = tuple(x.get_shape().as_list())[1:]
                            slice_i = get_slice(x, i, num_gpus)
                            _inputs.append(slice_i)
                    else:
                        _inputs.append(x)

                # Apply model on slice
                # (creating a model replica on the target device).
                outputs = model(_inputs)
                outputs = to_list(outputs)

                # Save the outputs for merging back together later.
                for o in range(len(outputs)):
                    all_outputs[o].append(outputs[o])

    # Merge outputs under expected scope.
    with tf.device('/cpu:0' if cpu_merge else '/gpu:%d' % target_gpu_ids[0]):
        merged = []
        for outputs in all_outputs:
            merged.append(tf.concat(outputs, axis=0))
    return merged

def build_admm_model(img, ATy, z0, Qindices, Qids, Qweights, Qshape,
                     rho, proj, clf_image, clf_latent=None, lat_img=None, max_iter=20, solver_tols=1e-2,
                     use_unroll_admm=False, lstsq=False):
    shape = img.get_shape().as_list()
    batch_size = shape[0]
    dim_img = shape[1] # 64
    dim_xz = np.prod(shape[1:]) # dim_img * dim_img * 3

    Qids_sp = tf.SparseTensor(Qindices, Qids, Qshape)
    Qweights_sp = tf.SparseTensor(Qindices, Qweights, Qshape)

    x = tf.zeros_like(z0)
    z = z0
    u = tf.zeros_like(z0)

    i = tf.constant(0, dtype=tf.int64)
    
    proj_img, lat_img = proj(img) # P(x), E(x), x \in [-1, 1]
    
    clf_img, _ = clf_image(img) # D(x)
    clf_proj_img, _ = clf_image(proj_img, reuse=True) # D(P(x))
    if clf_latent is not None:
        clf_lat_img, _ = clf_latent(lat_img) # D_l(E(x))
    else:
        clf_lat_img, _ = tf.zeros([batch_size], dtype=z0.dtype)
    
    clf_proj_noisy_img = tf.zeros((batch_size, 1), dtype=z0.dtype)
    clf_lat_noisy_img = tf.zeros((batch_size, 1), dtype=z0.dtype)
    
    cond = lambda i, x, z, u, clf_proj_noisy_img, clf_lat_noisy_img: tf.less(i, max_iter)
    def body(i, x, z, u, clf_proj_noisy_img, clf_lat_noisy_img):#x_z, 
        # everything shall be in [0, 1] here
        # except for proj input v, proj output P(v) and collected variables that are [-1 ,1]
        
        # x-update
        noisy_img = tf.reshape(z + u, (-1, dim_img, dim_img, 3))
        noisy_img = ((noisy_img - 0.5) * 2.0) # normalize to [-1, 1]
        
        proj_noisy_img_i, lat_noisy_img_i = proj(noisy_img, reuse=True) # P(v), E(v)
        
        clf_proj_noisy_img_i, _ = clf_image(proj_noisy_img_i, reuse=True) # D(P(v))
        if clf_latent is not None:
            clf_lat_noisy_img_i, _ = clf_latent(lat_noisy_img_i, reuse=True) # D_l(E(v))
        else:
            clf_lat_noisy_img_i = tf.zeros([batch_size], dtype=z0.dtype)
        
        x = tf.reshape(proj_noisy_img_i, (-1, dim_xz))
        x = ((x / 2.0) + 0.5) # normalize to [0, 1]
        
        clf_proj_noisy_img = tf.cond(tf.equal(i, 0), lambda: clf_proj_noisy_img_i[:, None],
                                     lambda: tf.concat([clf_proj_noisy_img, clf_proj_noisy_img_i[:, None]],  axis=1))
        clf_lat_noisy_img = tf.cond(tf.equal(i, 0), lambda: clf_lat_noisy_img_i[:, None],
                                    lambda: tf.concat([clf_lat_noisy_img, clf_lat_noisy_img_i[:, None]],  axis=1))
        
        # z-update
        b = ATy + rho * (x - u)
        z = tf.reshape(tf.transpose(tf.nn.embedding_lookup_sparse(tf.transpose(b), Qids_sp, Qweights_sp, combiner='sum')),
                       (batch_size, dim_xz))
        
        # u-update
        u = u + (z - x)

        i = i + 1

        return i, x, z, u, clf_proj_noisy_img, clf_lat_noisy_img #x_z, 
    shape_invariants = [i.get_shape(), tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                        tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None])]
    if use_unroll_admm is False:
        i, x, z, u, clf_proj_noisy_img, clf_lat_noisy_img = tf.while_loop(cond, body, [i, x, z, u, clf_proj_noisy_img,
                                                                                       clf_lat_noisy_img], shape_invariants)
    else:
        for _ in range(max_iter):
            i, x, z, u, clf_proj_noisy_img, clf_lat_noisy_img = body(i, x, z, u, clf_proj_noisy_img, clf_lat_noisy_img)

    x = tf.reshape(x, (-1, dim_img, dim_img, 3))
    z = tf.reshape(z, (-1, dim_img, dim_img, 3))
    u = tf.reshape(u, (-1, dim_img, dim_img, 3))
    x = ((x - 0.5) * 2.0) # normalize to [-1, 1]
    z = ((z - 0.5) * 2.0) # normalize to [-1, 1]
    u = ((u - 0.5) * 2.0) # normalize to [-1, 1]
    
    return x, z, u, proj_img, clf_img, clf_proj_img, clf_proj_noisy_img, clf_lat_img, clf_lat_noisy_img


def build_classifier_model_imagespace(image, is_train, n_reference, reuse=None, use_instance_norm=False, use_elu_like=False,
                                      use_custom_image_resize=False, dtype=tf.float32):
    """
    Build the graph for the classifier in the image space
    """

    channel_compress_ratio = 4

    with tf.variable_scope('DIS', reuse=reuse):

        with tf.variable_scope('IMG'):
            ## image space D
            # 1
            conv1 = layers.new_conv_layer(image, [4,4,3,64], stride=1, name="conv1",
                                          use_custom_image_resize=use_custom_image_resize, dtype=dtype) #64

            # 2
            nBlocks = 3
            module2 = layers.add_bottleneck_module(conv1, is_train, nBlocks, n_reference,
                                                   channel_compress_ratio=channel_compress_ratio, name='module2',
                                                   use_instance_norm=use_instance_norm, use_elu_like=use_elu_like,
                                                   use_custom_image_resize=use_custom_image_resize, dtype=dtype) # 32

            # 3
            nBlocks = 4
            module3 = layers.add_bottleneck_module(module2, is_train, nBlocks, n_reference,
                                                   channel_compress_ratio=channel_compress_ratio, name='module3',
                                                   use_instance_norm=use_instance_norm, use_elu_like=use_elu_like,
                                                   use_custom_image_resize=use_custom_image_resize, dtype=dtype) # 16

            # 4
            nBlocks = 6
            module4 = layers.add_bottleneck_module(module3, is_train, nBlocks, n_reference,
                                                   channel_compress_ratio=channel_compress_ratio, name='module4',
                                                   use_instance_norm=use_instance_norm, use_elu_like=use_elu_like,
                                                   use_custom_image_resize=use_custom_image_resize, dtype=dtype) # 8

            # 5
            nBlocks = 3
            module5 = layers.add_bottleneck_module(module4, is_train, nBlocks, n_reference,
                                                   channel_compress_ratio=channel_compress_ratio, name='module5',
                                                   use_instance_norm=use_instance_norm, use_elu_like=use_elu_like,
                                                   use_custom_image_resize=use_custom_image_resize, dtype=dtype) # 4
            if use_instance_norm is False:
                tmp = layers.batchnorm(module5, is_train, n_reference, name='bn_module5')
            else:
                tmp = layers.instancenorm(module5, is_train, n_reference, name='bn_module5')
            if use_elu_like is False:
                bn_module5 = tf.nn.elu(tmp)
            else:
                bn_module5 = layers.nn_elu_like(tmp)
            
            (dis, last_w) = layers.new_fc_layer(bn_module5, output_size=1, name='dis', dtype=dtype)

    return dis[:,0], last_w


def build_classifier_model_latentspace(latent, is_train, n_reference, reuse=None, use_instance_norm=False, use_elu_like=False,
                                       use_custom_image_resize=False, dtype=tf.float32):
    """
    Build the graph for the classifier in the latent space
    """

    channel_compress_ratio = 4

    with tf.variable_scope('DIS', reuse=reuse):

        with tf.variable_scope('LATENT'):

            out = layers.bottleneck(latent, is_train, n_reference,
                                    channel_compress_ratio=channel_compress_ratio, stride=1, name='block0',
                                    use_instance_norm=use_instance_norm, use_elu_like=use_elu_like,
                                    use_custom_image_resize=use_custom_image_resize, dtype=dtype) # 8*8*4096
            out = layers.bottleneck(out, is_train, n_reference,
                                    channel_compress_ratio=channel_compress_ratio, stride=1, name='block1',
                                    use_instance_norm=use_instance_norm, use_elu_like=use_elu_like,
                                    use_custom_image_resize=use_custom_image_resize, dtype=dtype) # 8*8*4096
            out = layers.bottleneck(out, is_train, n_reference,
                                    channel_compress_ratio=channel_compress_ratio, stride=1, name='block2',
                                    use_instance_norm=use_instance_norm, use_elu_like=use_elu_like,
                                    use_custom_image_resize=use_custom_image_resize, dtype=dtype) # 8*8*4096

            output_channel = out.get_shape().as_list()[-1]
            out = layers.bottleneck_flexible(out, is_train, output_channel, n_reference,
                                             channel_compress_ratio=4, stride=2, name='block3',
                                             use_instance_norm=use_instance_norm, use_elu_like=use_elu_like,
                                             use_custom_image_resize=use_custom_image_resize, dtype=dtype) # 4*4*4096
            out = layers.bottleneck(out, is_train, n_reference,
                                    channel_compress_ratio=4, stride=1, name='block4',
                                    use_instance_norm=use_instance_norm, use_elu_like=use_elu_like,
                                    use_custom_image_resize=use_custom_image_resize, dtype=dtype) # 4*4*4096
            out = layers.bottleneck(out, is_train, n_reference,
                                    channel_compress_ratio=4, stride=1, name='block5',
                                    use_instance_norm=use_instance_norm, use_elu_like=use_elu_like,
                                    use_custom_image_resize=use_custom_image_resize, dtype=dtype) # 4*4*4096
            if use_instance_norm is False:
                tmp = layers.batchnorm(out, is_train, n_reference, name='bn1')
            else:
                tmp = layers.instancenorm(out, is_train, n_reference, name='bn1')
            if use_elu_like is False:
                bn1 = tf.nn.elu(tmp)
            else:
                bn1 = layers.nn_elu_like(tmp)
            (dis, last_w) = layers.new_fc_layer(bn1, output_size=1, name='dis', dtype=dtype)

    return dis[:,0], last_w


def build_projection_model(images, is_train, n_reference, use_bias=True, reuse=None, use_instance_norm=False, use_elu_like=False,
                           use_custom_image_resize=False, dtype=tf.float32):
    """
    Build the graph for the projection network, which shares the architecture of a typical autoencoder.
    To improve contextual awareness, we add a channel-wise fully-connected layer followed by a 2-by-2
    convolution layer at the middle.
    """
    channel_compress_ratio = 4
    dim_latent = 1024

    with tf.variable_scope('PROJ', reuse=reuse):

        with tf.variable_scope('ENCODE'):
            conv0 = layers.new_conv_layer(images, [4,4,3,64], stride=1, bias=use_bias, name="conv0",
                                          use_custom_image_resize=use_custom_image_resize, dtype=dtype ) #64
            if use_instance_norm is False:
                tmp = layers.batchnorm(conv0, is_train, n_reference, name='bn0')
            else:
                tmp = layers.instancenorm(conv0, is_train, n_reference, name='bn0')
            if use_elu_like is False:
                bn0 = tf.nn.elu(tmp)
            else:
                bn0 = layers.nn_elu_like(tmp)
            conv1 = layers.new_conv_layer(bn0, [4,4,64,128], stride=1, bias=use_bias, name="conv1",
                                          use_custom_image_resize=use_custom_image_resize, dtype=dtype ) #64
            if use_instance_norm is False:
                tmp = layers.batchnorm(conv1, is_train, n_reference, name='bn1')
            else:
                tmp = layers.instancenorm(conv1, is_train, n_reference, name='bn1')
            if use_elu_like is False:
                bn1 = tf.nn.elu(tmp)
            else:
                bn1 = layers.nn_elu_like(tmp)
            conv2 = layers.new_conv_layer(bn1, [4,4,128,256], stride=2, bias=use_bias, name="conv2",
                                          use_custom_image_resize=use_custom_image_resize, dtype=dtype) #32
            if use_instance_norm is False:
                tmp = layers.batchnorm(conv2, is_train, n_reference, name='bn2')
            else:
                tmp = layers.instancenorm(conv2, is_train, n_reference, name='bn2')
            if use_elu_like is False:
                bn2 = tf.nn.elu(tmp)
            else:
                bn2 = layers.nn_elu_like(tmp)
            conv3 = layers.new_conv_layer(bn2, [4,4,256,512], stride=2, bias=use_bias, name="conv3",
                                          use_custom_image_resize=use_custom_image_resize, dtype=dtype) #16
            if use_instance_norm is False:
                tmp = layers.batchnorm(conv3, is_train, n_reference, name='bn3')
            else:
                tmp = layers.instancenorm(conv3, is_train, n_reference, name='bn3')
            if use_elu_like is False:
                bn3 = tf.nn.elu(tmp)
            else:
                bn3 = layers.nn_elu_like(tmp)
            conv4 = layers.new_conv_layer(bn3, [4,4,512,dim_latent], stride=2, bias=use_bias, name="conv4",
                                          use_custom_image_resize=use_custom_image_resize, dtype=dtype) #8
            if use_instance_norm is False:
                tmp = layers.batchnorm(conv4, is_train, n_reference, name='bn4')
            else:
                tmp = layers.instancenorm(conv4, is_train, n_reference, name='bn4')
            if use_elu_like is False:
                bn4 = tf.nn.elu(tmp)
            else:
                bn4 = layers.nn_elu_like(tmp)
            fc5 = layers.channel_wise_fc_layer(bn4, 'fc5', bias=False, dtype=dtype)
            fc5_conv = layers.new_conv_layer(fc5, [2,2,dim_latent, dim_latent], stride=1, bias=use_bias, name="conv_fc",
                                             use_custom_image_resize=use_custom_image_resize, dtype=dtype)
            if use_instance_norm is False:
                tmp = layers.batchnorm(fc5_conv, is_train, n_reference, name='latent')
            else:
                tmp = layers.instancenorm(fc5_conv, is_train, n_reference, name='latent')
            if use_elu_like is False:
                latent = tf.nn.elu(tmp)
            else:
                latent = layers.nn_elu_like(tmp)


        deconv3 = layers.new_deconv_layer(latent, [4,4,512,dim_latent], conv3.get_shape().as_list(), stride=2, bias=use_bias, name="deconv3",
                                          use_custom_image_resize=use_custom_image_resize, dtype=dtype)
        if use_instance_norm is False:
            tmp = layers.batchnorm(deconv3, is_train, n_reference, name='debn3')
        else:
            tmp = layers.instancenorm(deconv3, is_train, n_reference, name='debn3')
        if use_elu_like is False:
            debn3 = tf.nn.elu(tmp)
        else:
            debn3 = layers.nn_elu_like(tmp)
        deconv2 = layers.new_deconv_layer(debn3, [4,4,256,512], conv2.get_shape().as_list(), stride=2, bias=use_bias, name="deconv2",
                                          use_custom_image_resize=use_custom_image_resize, dtype=dtype)
        if use_instance_norm is False:
            tmp = layers.batchnorm(deconv2, is_train, n_reference, name='debn2')
        else:
            tmp = layers.instancenorm(deconv2, is_train, n_reference, name='debn2')
        if use_elu_like is False:
            debn2 = tf.nn.elu(tmp)
        else:
            debn2 = layers.nn_elu_like(tmp)
        deconv1 = layers.new_deconv_layer(debn2, [4,4,128,256], conv1.get_shape().as_list(), stride=2, bias=use_bias, name="deconv1",
                                          use_custom_image_resize=use_custom_image_resize, dtype=dtype)
        if use_instance_norm is False:
            tmp = layers.batchnorm(deconv1, is_train, n_reference, name='debn1')
        else:
            tmp = layers.instancenorm(deconv1, is_train, n_reference, name='debn1')
        if use_elu_like is False:
            debn1 = tf.nn.elu(tmp)
        else:
            debn1 = layers.nn_elu_like(tmp)
        deconv0 = layers.new_deconv_layer(debn1, [4,4,64,128], conv0.get_shape().as_list(), stride=1, bias=use_bias, name="deconv0",
                                          use_custom_image_resize=use_custom_image_resize, dtype=dtype)
        if use_instance_norm is False:
            tmp = layers.batchnorm(deconv0, is_train, n_reference, name='debn0')
        else:
            tmp = layers.instancenorm(deconv0, is_train, n_reference, name='debn0')
        if use_elu_like is False:
            debn0 = tf.nn.elu(tmp)
        else:
            debn0 = layers.nn_elu_like(tmp)
        proj_ori = layers.new_deconv_layer(debn0, [4,4,3,64], images.get_shape().as_list(), stride=1, bias=use_bias, name="recon",
                                           use_custom_image_resize=use_custom_image_resize, dtype=dtype)
        proj = proj_ori

    return proj, latent


if __name__ == '__main__':

    ### parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', default=None, help='Where to store samples and models')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--img_size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--n_reference', type=int, default=32, help='the size of reference batch')
    parser.add_argument('--Dperiod', type=int, default=1, help='number of continuous D update')
    parser.add_argument('--Gperiod', type=int, default=1, help='number of continuous G update')

    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--pretrained_iter', type=int, default=0, help='iter of the pretrained model, if 0 then not using')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')

    parser.add_argument('--learning_rate_val_proj', type=float, default=0.002, help='learning rate, default=0.002')
    parser.add_argument('--learning_rate_val_dis', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--weight_decay_rate', type=float, default=0.00001, help='weight decay rate, default=0.00000')
    parser.add_argument('--clamp_weight', type=int, default=1)
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)

    parser.add_argument('--use_spatially_varying_uniform_on_top', type=int, default=1, help='Whether to multiply the gaussian noise with a uniform noise map to avoid overfitting')

    parser.add_argument('--continuous_noise', type=int, default=1, help='whether to use continuous noise_std ')
    parser.add_argument('--noise_std', type=float, default=1.2, help='std of the added noise, default = 1.2')

    parser.add_argument('--uniform_noise_max', type=float, default=3.464, help='The range of the uniform noise, default = 3.464 to make overall std remain unchange')
    parser.add_argument('--min_spatially_continuous_noise_factor', type=float, default=0.01, help='The lower the value, the higher the possibility the varying of the noise be more continuous')
    parser.add_argument('--max_spatially_continuous_noise_factor', type=float, default=0.5, help='The upper the value, the higher the possibility the varying of the noise be more continuous')
    parser.add_argument('--adam_beta1_d', type=float, default=0.9, help='beta1 of adam for the critic, default = 0.9')
    parser.add_argument('--adam_beta2_d', type=float, default=0.999, help='beta2 of adam for the critic, default = 0.999')
    parser.add_argument('--adam_eps_d', type=float, default=1e-8, help='eps of adam for the critic, default = 1e-8')
    parser.add_argument('--adam_beta1_g', type=float, default=0.9, help='beta1 of adam for the projector, default = 0.9')
    parser.add_argument('--adam_beta2_g', type=float, default=0.999, help='beta2 of adam for the projector, default = 0.999')
    parser.add_argument('--adam_eps_g', type=float, default=1e-5, help='eps of adam for the projector, default = 1e-8')

    parser.add_argument('--use_tensorboard', type=int, default=1, help='whether to use tensorboard')
    parser.add_argument('--tensorboard_period', type=int, default=1, help='how often to write to tensorboard')
    parser.add_argument('--output_img', type=int, default=0, help='whether to output images, (also act as the number of images to output)')
    parser.add_argument('--output_img_period', type=int, default=100, help='how often to output images')

    parser.add_argument('--clip_input', type=int, default=0, help='clip the input to the network')
    parser.add_argument('--clip_input_bound', type=float, default=2.0, help='the maximum of input')

    parser.add_argument('--lambda_ratio', type=float, default=1e-2, help='the weight ratio in the objective function of true images to fake images, default 1e-2')
    parser.add_argument('--lambda_l2', type=float, default=5e-3, help='lambda of l2 loss, default = 5e-3')
    parser.add_argument('--lambda_latent', type=float, default=1e-4, help='lambda of latent adv loss, default = 1e-4')
    parser.add_argument('--lambda_img', type=float, default=1e-3, help='lambda of img adv loss, default = 1e-3')
    parser.add_argument('--lambda_de', type=float, default=1.0, help='lambda of the denoising autoencoder, default = 1.0')
    parser.add_argument('--de_decay_rate', type=float, default=1.0, help='the rate lambda_de decays, default = 1.0')

    parser.add_argument('--one_sided_label_smooth', type=float, default=0.85, help='the positive label for one-sided, default = 0.85')

    parser.add_argument('--use_instance_norm', type=int, default=0, help='whether to use instance normalization instead of the outdated batch normalization variant')
    parser.add_argument('--use_elu_like', type=int, default=0, help='whether to use elu-like activation')
    parser.add_argument('--use_custom_image_resize', type=int, default=0, help='whether to use custom implemented differentiable image resizing')
    parser.add_argument('--use_diff_admm', type=int, default=0, help='whether to use backpropagation across admm iterations')
    parser.add_argument('--use_unroll_admm', type=int, default=0, help='whether to use unrolled admm iterations')
    parser.add_argument('--n_admm_iters', type=int, default=8, help='admm iteration count')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpu-s used for training')
    parser.add_argument('--data_set', default='imagenet', help='which dataset to use, imagenet or celeb')
        
    opt = parser.parse_args()
    print(opt)

    ### parameters ###
    n_epochs = int(opt.n_epochs)
    learning_rate_val_dis = float(opt.learning_rate_val_dis) # 0.0004 #
    learning_rate_val_proj = float(opt.learning_rate_val_proj) # 0.0001 #
    learning_rate_val_proj_max = learning_rate_val_proj
    learning_rate_val_proj_current = learning_rate_val_proj
    weight_decay_rate =  float(opt.weight_decay_rate)

    std = float(opt.noise_std)
    continuous_noise = int(opt.continuous_noise)

    use_spatially_varying_uniform_on_top = int(opt.use_spatially_varying_uniform_on_top)
    uniform_noise_max = float(opt.uniform_noise_max)
    min_spatially_continuous_noise_factor = float(opt.min_spatially_continuous_noise_factor)
    max_spatially_continuous_noise_factor = float(opt.max_spatially_continuous_noise_factor)

    img_size = int(opt.img_size)
    Dperiod = int(opt.Dperiod) # 1 #
    Gperiod = int(opt.Gperiod)

    clamp_weight = int(opt.clamp_weight)
    clamp_lower = float(opt.clamp_lower)
    clamp_upper = float(opt.clamp_upper)

    random_seed = int(opt.random_seed)

    adam_beta1_d = float(opt.adam_beta1_d) # 0. #
    adam_beta2_d = float(opt.adam_beta2_d) # 0.9 #
    adam_eps_d = float(opt.adam_eps_d)

    adam_beta1_g = float(opt.adam_beta1_g) # 0. #
    adam_beta2_g = float(opt.adam_beta2_g) # 0.9 #
    adam_eps_g = float(opt.adam_eps_g)

    use_tensorboard = int(opt.use_tensorboard)
    tensorboard_period = int(opt.tensorboard_period)
    output_img = int(opt.output_img)
    output_img_period = int(opt.output_img_period)

    clip_input = int(opt.clip_input)
    clip_input_bound = float(opt.clip_input_bound)

    lambda_ratio = float(opt.lambda_ratio)
    lambda_l2 = float(opt.lambda_l2)
    lambda_latent = float(opt.lambda_latent)
    lambda_img = float(opt.lambda_img)
    lambda_de = float(opt.lambda_de)
    de_decay_rate = float(opt.de_decay_rate)

    one_sided_label_smooth = float(opt.one_sided_label_smooth)

    use_instance_norm = int(opt.use_instance_norm)
    use_elu_like = int(opt.use_elu_like)
    use_custom_image_resize = int(opt.use_custom_image_resize)
    use_diff_admm = int(opt.use_diff_admm)
    use_unroll_admm = int(opt.use_unroll_admm)
    n_admm_iters = int(opt.n_admm_iters)
    gpus = int(opt.gpus)

    data_set = opt.data_set
    if data_set not in ['imagenet', 'celeb']:
        raise NotImplementedError
    if data_set == 'imagenet':
        print('Loading ImageNet data set...')
        import load_imagenet as load_dataset
    elif data_set == 'celeb':
        print('Loading MS-Celeb-1M data set...')
        import load_celeb as load_dataset

    dtype = tf.float32
    batch_size = int(opt.batch_size)
    n_reference = int(opt.n_reference)
    #batch_size=5
    inst_size = batch_size - n_reference
    """
    if use_instance_norm is False:
        batch_size = int(opt.batch_size)
        n_reference = int(opt.n_reference)
    else:
        batch_size = int(opt.batch_size) // 2
        n_reference = 0
    inst_size = batch_size - n_reference
    """

    base_folder = opt.base_folder

    if base_folder == None:
        base_folder = 'model'

    dt = datetime.datetime.now()
    dtime_pid = '{y:04d}{mo:02d}{d:02d}{h:02d}{mi:02d}{s:02d}_{p:05d}'.format(y=dt.year, mo=dt.month, d=dt.day, h=dt.hour, mi=dt.minute, s=dt.second, p=os.getpid()) + '_' + os.uname()[1]

    base_folder = '%s/%s_imsize%d_ratio%f_dis%f_latent%f_img%f_de%f_derate%f_dp%d_gd%d_softpos%f_wdcy_%f_seed%d' % (
        base_folder, dtime_pid, img_size, lambda_ratio, lambda_l2, lambda_latent, lambda_img,
        lambda_de, de_decay_rate,
        Dperiod, Gperiod,
        one_sided_label_smooth,
        weight_decay_rate,
        random_seed
    )


    model_path = '%s/model' % (base_folder)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    epoch_path = '%s/epoch' % (base_folder)
    if not os.path.exists(epoch_path):
        os.makedirs(epoch_path)

    init_path = '%s/init' % (base_folder)
    if not os.path.exists(init_path):
        os.makedirs(init_path)

    img_path = '%s/image' % (base_folder)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    logs_base = '/tmp/tensorflow_logs'
    logs_path = '%s/%s' % (logs_base, base_folder)

    # write configurations to a file
    filename = '%s/configurations.txt' % (base_folder)
    f = open( filename, 'a' )
    f.write( repr(opt) + '\n' )
    f.close()

    pretrained_iter = int(opt.pretrained_iter)
    use_pretrain = pretrained_iter > 0
    pretrained_model_file = '%s/model_iter-%d' % (model_path, pretrained_iter)

    tf.set_random_seed(random_seed)


    ### load the dataset ###
    def read_file_cpu(trainset, queue, batch_size, num_prepare, use_diff_admm=False, dtype=np.float32, rseed=None):
        local_random = np.random.RandomState(rseed)
        n_train = len(trainset)
        trainset_index = local_random.permutation(n_train)
        idx = 0
        
        while True:
            # read in data if the queue is too short
            while queue.full() == False:
                batch = np.zeros([batch_size, img_size, img_size, 3], np.float32)
                if use_diff_admm is False:
                    noisy_batch = np.zeros([batch_size, img_size, img_size, 3], np.float32)
                else:
                    r = np.random.randint(4)
                    if r == 0:
                        (A_fun, AT_fun, mask, A) = problem_inpaint.setup((1, img_size, img_size, 3), drop_prob=0.5)
                        rho = 0.3
                    elif r == 1:
                        (A_fun, AT_fun, mask, A) = problem_inpaint_center.setup((1, img_size, img_size, 3), box_size=int(0.3 * img_size))
                        rho = 0.2
                    elif r == 2:
                        (A_fun, AT_fun, mask, A) = problem_inpaint_block.setup((1, img_size, img_size, 3), box_size=int(0.1 * img_size), total_box=10)
                        rho = 0.3
                    elif r == 3:
                        (A_fun, AT_fun, A) = problem_superres.setup((1, img_size, img_size, 3), resize_ratio=0.5)
                        rho = 0.5
                    ATy_batch = np.zeros([batch_size, A.shape[1]], np.float32)
                    z0_batch = np.zeros([batch_size, A.shape[1]], np.float32)
                    ATArho = (A.T.dot(A) + rho * scipy.sparse.eye(A.shape[1], format='csc', dtype=np.float32)).tocsc()
                    Q = scipy.sparse.linalg.inv(ATArho)
                    print(r, A.nnz / np.prod(A.shape), ATArho.nnz / np.prod(ATArho.shape), Q.nnz / np.prod(Q.shape))
                    Qindices = np.asarray([[row_i, col_i]
                                          for row_i, nnz in enumerate(np.diff(Q.indptr))
                                          for col_i in range(nnz)], dtype=np.int64)
                    Qids = Q.indices.astype(np.int64)
                    Qweights = Q.data        
                    Qshape = Q.shape                
                for i in range(batch_size):
                    image_path = trainset[trainset_index[idx+i]]
                    img = imread(image_path)
                    # <Note> In our original code used to generate the results in the paper, we directly
                    # resize the image directly to the input dimension via (for both ms-celeb-1m and imagenet)
                    img = imresize(img, [img_size, img_size]).astype(float) / 255.0
                    
                    # The following code crops random-sized patches (may be useful for imagenet)
                    #img_shape = img.shape
                    #min_edge = min(img_shape[0], img_shape[1])
                    #min_resize_ratio = float(img_size) / float(min_edge)
                    #max_resize_ratio = min_resize_ratio * 2.0
                    #resize_ratio = local_random.rand() * (max_resize_ratio - min_resize_ratio) + min_resize_ratio

                    #img = sp.misc.imresize(img, resize_ratio).astype(float) / 255.0
                    #crop_loc_row = local_random.randint(img.shape[0]-img_size+1)
                    #crop_loc_col = local_random.randint(img.shape[1]-img_size+1)
                    #if len(img.shape) == 3:
                        #img = img[crop_loc_row:crop_loc_row+img_size, crop_loc_col:crop_loc_col+img_size,:]
                    #else:
                        #img = img[crop_loc_row:crop_loc_row+img_size, crop_loc_col:crop_loc_col+img_size]

                    if np.prod(img.shape) == 0:
                        img = np.zeros([img_size, img_size, 3], dtype=np.float32)

                    if len(img.shape) < 3:
                        img = np.expand_dims(img, axis=2)
                        img = np.tile(img, [1,1,3])

                    ## random flip
                    #flip_prob = local_random.rand()
                    #if flip_prob < 0.5:
                        #img = img[-1:None:-1,:,:]

                    #flip_prob = local_random.rand()
                    #if flip_prob < 0.5:
                        #img = img[:,-1:None:-1,:]

                    # add noise to img
                    if use_diff_admm is False:
                        noisy_img = add_noise(img, local_random,
                                std=std,
                                uniform_max=uniform_noise_max,
                                min_spatially_continuous_noise_factor=min_spatially_continuous_noise_factor,
                                max_spatially_continuous_noise_factor=max_spatially_continuous_noise_factor,
                                continuous_noise=continuous_noise,
                                use_spatially_varying_uniform_on_top=use_spatially_varying_uniform_on_top,
                                clip_input=clip_input, clip_input_bound=clip_input_bound
                                )

                    batch[i] = img.astype(np.float32)
                    if use_diff_admm is False:
                        noisy_batch[i] = noisy_img.astype(np.float32) # add generated noisy image to batch here
                    else:
                        y, _ = add_noise_admm.exe(A_fun(img[None]), noise_mean=0.0, noise_std=0.1 if r == 0 else 0.0)
                        y = y[0].reshape(1, -1).astype(np.float32)
                        ATy = ((A.T.dot(y.T)).T)
                        ATy_batch[i] = ATy
                        z0 = scipy.sparse.linalg.lsmr(A, y.flatten())[0]
                        z0_batch[i] = z0

                batch *= 2.0
                batch -= 1.0
                if use_diff_admm is False:
                    noisy_batch *= 2.0
                    noisy_batch -= 1.0
                        

                if clip_input > 0:
                    batch = np.clip(batch, a_min=-clip_input_bound, a_max=clip_input_bound)
                    if use_diff_admm is False:
                        noisy_batch = np.clip(noisy_batch, a_min=-clip_input_bound, a_max=clip_input_bound)

                if use_diff_admm is False:
                    queue.put([batch, noisy_batch]) # block until free slot is available
                else:
                    queue.put([batch, ATy_batch, z0_batch, Qindices, Qids, Qweights, Qshape, rho]) # block until free slot is available                        

                idx += batch_size
                if idx > n_train: #reset when last batch is smaller than batch_size or reaching the last batch
                    trainset_index = local_random.permutation(n_train)
                    idx = 0

    def create_train_procs(trainset, train_queue, n_thread, num_prepare, train_procs, use_diff_admm=False, dtype=tf.float32):
        """
        create threads to read the images from hard drive and perturb them
        """
        for n_read in range(n_thread):
            seed = np.random.randint(1e8)
            instance_size = batch_size - n_reference
            if instance_size < 1:
                print('ERROR: batch_size < n_reference + 1')
            train_proc = Process(target=read_file_cpu, args=(trainset, train_queue, instance_size, num_prepare, use_diff_admm, dtype, seed))
            train_proc.daemon = True
            train_proc.start()
            train_procs.append(train_proc)

    def terminate_train_procs(train_procs):
        """
        terminate the threads to force garbage collection and free memory
        """
        for procs in train_procs:
            procs.terminate()


    trainset = load_dataset.load_trainset_path_list()
    total_train = len(trainset)
    print('total train = %d' % (total_train))


    if use_instance_norm is False:
        print("create reference batch...")
        n_thread = 1
        num_prepare = 1
        reference_queue = Queue(num_prepare)
        ref_seed = 1085 # the random seed particularly for creating the reference batch
        ref_proc = Process(target=read_file_cpu, args=(trainset, reference_queue, n_reference, num_prepare, use_diff_admm, dtype, ref_seed))
        ref_proc.daemon = True
        ref_proc.start()

        ref_batch, ref_noisy_batch = reference_queue.get()
        
        ref_proc.terminate()
        del ref_proc
        del reference_queue

        # save reference to a mat file
        ref_file = '%s/ref_batch_%d.mat' % (base_folder, n_reference)
        sp.io.savemat(ref_file, {'ref_batch': ref_batch})
        print('ref_batch saved.')

    def np_combine_batch(inst,ref):
        out = np.concatenate([inst,ref], axis=0)
        return out
    def get_inst(batch):
        return batch[0:inst_size]
            


    print("loading data...")

    n_thread = 16
    num_prepare = 20
    print('total train = %d' % (total_train))
    train_queue = Queue(num_prepare+1)
    train_procs = []
    create_train_procs(trainset, train_queue, n_thread, num_prepare, train_procs, use_diff_admm, dtype)


    ### set up the graph
    def model(inputs, dtype=tf.float32, reuse=None):
        
        if use_diff_admm is False:
            images_tf, noisy_image_tf = inputs

            # build autoencoder
            projection_x_all, latent_x_all = build_projection_model(images_tf, is_train, n_reference, use_instance_norm=use_instance_norm, use_elu_like=use_elu_like, use_custom_image_resize=use_custom_image_resize, dtype=dtype, reuse=reuse)
            projection_x = get_inst(projection_x_all) # P(x)
            latent_x = get_inst(latent_x_all) # E(x)

            projection_z_all, latent_z_all = build_projection_model(noisy_image_tf, is_train, n_reference, use_instance_norm=use_instance_norm, use_elu_like=use_elu_like, use_custom_image_resize=use_custom_image_resize, dtype=dtype, reuse=True)
            projection_z = get_inst(projection_z_all) # P(v)
            latent_z = get_inst(latent_z_all) # E(v)

            # build the discriminator
            # image space
            adversarial_truex_all, _ = build_classifier_model_imagespace(images_tf, is_train, n_reference, use_instance_norm=use_instance_norm, use_elu_like=use_elu_like, use_custom_image_resize=use_custom_image_resize, dtype=dtype, reuse=reuse)
            adversarial_truex = get_inst(adversarial_truex_all) # D(x)

            adversarial_projx_all, _ = build_classifier_model_imagespace(projection_x_all, is_train, n_reference, use_instance_norm=use_instance_norm, use_elu_like=use_elu_like, use_custom_image_resize=use_custom_image_resize, dtype=dtype, reuse=True)
            adversarial_projx = get_inst(adversarial_projx_all) # D(P(x))

            adversarial_projz_all, _ = build_classifier_model_imagespace(projection_z_all, is_train, n_reference, use_instance_norm=use_instance_norm, use_elu_like=use_elu_like, use_custom_image_resize=use_custom_image_resize, dtype=dtype, reuse=True)
            adversarial_projz = get_inst(adversarial_projz_all) # D(P(v))

            # latent space
            if lambda_latent > 0:
                adversarial_latentx_all, _ = build_classifier_model_latentspace(latent_x_all, is_train, n_reference, use_instance_norm=use_instance_norm, use_elu_like=use_elu_like, use_custom_image_resize=use_custom_image_resize, dtype=dtype, reuse=reuse)
                adversarial_latentz_all, _ = build_classifier_model_latentspace(latent_z_all, is_train, n_reference, use_instance_norm=use_instance_norm, use_elu_like=use_elu_like, use_custom_image_resize=use_custom_image_resize, dtype=dtype, reuse=True)
            else:
                adversarial_latentx_all = tf.zeros([batch_size], dtype=dtype)
                adversarial_latentz_all = tf.zeros([batch_size], dtype=dtype)

            adversarial_latentx = get_inst(adversarial_latentx_all) # D_l(E(x))
            adversarial_latentz = get_inst(adversarial_latentz_all) # D_l(E(v))
            outputs = [projection_x, projection_z, adversarial_truex, adversarial_projx, adversarial_projz, adversarial_latentx, adversarial_latentz]#, projection_z]
        else:
            images_tf, ATy_tf, z0_tf, Qindices_tf, Qids_tf, Qweights_tf, Qshape_tf, rho_tf = inputs

            x_all, z_all, u_all, projection_x_all, adversarial_truex_all, adversarial_projx_all, \
                adversarial_projz_all, adversarial_latentx_all, adversarial_latentz_all = \
                    build_admm_model(images_tf, ATy_tf, z0_tf, Qindices_tf, Qids_tf, Qweights_tf, Qshape_tf, rho_tf,
                                     lambda x, reuse=reuse: build_projection_model(x, is_train, n_reference, use_instance_norm=use_instance_norm, use_elu_like=use_elu_like, use_custom_image_resize=use_custom_image_resize, dtype=dtype, reuse=reuse),
                                     lambda x, reuse=reuse: build_classifier_model_imagespace(x, is_train, n_reference, use_instance_norm=use_instance_norm, use_elu_like=use_elu_like, use_custom_image_resize=use_custom_image_resize, dtype=dtype, reuse=reuse),
                                     lambda latent, reuse=reuse: build_classifier_model_latentspace(latent, is_train, n_reference, use_instance_norm=use_instance_norm, use_elu_like=use_elu_like, use_custom_image_resize=use_custom_image_resize, dtype=dtype, reuse=reuse),
                                     max_iter=n_admm_iters, use_unroll_admm=use_unroll_admm)
            x = get_inst(x_all) # x
            z = get_inst(z_all) # v
            u = get_inst(u_all) # u
            projection_x = get_inst(projection_x_all) # P(x)
            adversarial_truex = get_inst(adversarial_truex_all) # D(x)
            adversarial_projx = get_inst(adversarial_projx_all) # D(P(x)
            adversarial_projz = get_inst(adversarial_projz_all) # D(P(v))
            adversarial_latentx = get_inst(adversarial_latentx_all) # D_l(E(x))
            adversarial_latentz = get_inst(adversarial_latentz_all) # D_l(E(v))
            outputs = [x, z, u, projection_x, adversarial_truex, adversarial_projx, adversarial_projz, adversarial_latentx, adversarial_latentz]
        
        _outputs = []
        for o in outputs:
            _outputs.append(tf.cast(o, tf.float32))
        return _outputs

    is_train = True
    with tf.variable_scope('fp32_storage', custom_getter=float32_variable_storage_getter): # Note: This forces trainable variables to be stored as float32
        with tf.device('/cpu:0'):
            # images
            ATy_tf = tf.placeholder( dtype, [batch_size, img_size * img_size * 3], name="ATy_tf")
            z0_tf = tf.placeholder( dtype, [batch_size, img_size * img_size * 3], name="z0_tf")
            Qindices_tf = tf.placeholder(tf.int64, [None, 2])
            Qids_tf = tf.placeholder(tf.int64, [None])
            Qweights_tf = tf.placeholder(dtype, [None])
            Qshape_tf = tf.placeholder(tf.int64, [2])
            rho_tf = tf.placeholder( dtype, [], name="rho_tf")
            images_tf = tf.placeholder( dtype, [batch_size, img_size, img_size, 3], name="images_tf")
            images_tf_float32 = tf.cast(images_tf, tf.float32)
            noisy_image_tf = tf.placeholder( dtype, [batch_size, img_size, img_size, 3], name="noisy_image_tf")
            noisy_image_tf_float32 = tf.cast(noisy_image_tf, tf.float32)

            # lambdas
            lambda_ratio_tf = tf.placeholder( tf.float32, [], name='lambda_ratio_tf')
            lambda_l2_tf = tf.placeholder( tf.float32, [], name='lambda_l2_tf')
            lambda_latent_tf = tf.placeholder( tf.float32, [], name='lambda_latent_tf')
            lambda_img_tf = tf.placeholder( tf.float32, [], name='lambda_img')
            lambda_de_tf = tf.placeholder( tf.float32, [], name='lambda_de')

            learning_rate_dis = tf.placeholder( tf.float32, [], name='learning_rate_dis')
            learning_rate_proj = tf.placeholder( tf.float32, [], name='learning_rate_proj')
            adam_beta1_d_tf = tf.placeholder( tf.float32, [], name='adam_beta1_d_tf')
            adam_beta1_g_tf = tf.placeholder( tf.float32, [], name='adam_beta1_g_tf')

            if use_diff_admm is False:
                inputs = [images_tf, noisy_image_tf]
                slice_input = [True, True]
            else:
                inputs = [images_tf, ATy_tf, z0_tf, Qindices_tf, Qids_tf, Qweights_tf, Qshape_tf, rho_tf]
                slice_input = [True, True, True, False, False, False, False, False]
            outputs = model(inputs, dtype=dtype)
        if gpus > 1:
            multi_gpu_outputs = multi_gpu_model(inputs, slice_input, len(outputs), lambda inputs: model(inputs, dtype=dtype, reuse=True), gpus=gpus)
        else:
            multi_gpu_outputs = outputs
        if use_diff_admm is False:
            projection_x, projection_z, adversarial_truex, adversarial_projx, adversarial_projz, adversarial_latentx, adversarial_latentz = multi_gpu_outputs
        else:
            x, z, u, projection_x, adversarial_truex, adversarial_projx, adversarial_projz, adversarial_latentx, adversarial_latentz = multi_gpu_outputs

        # update_op for batch_norm moving average
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
        else:
            print('something is wrong!')
            updates = tf.zeros([1])
        
        # if we are using virtual batch normalization, we do not need to calculate the population mean and variance
        if n_reference > 0:
            updates = tf.zeros([1])
        # set up the loss
        pos_labels = tf.ones([inst_size], dtype=tf.float32)#1)
        if use_diff_admm is False:
            pos_labels_iter = pos_labels
        else:
            pos_labels_iter = tf.ones([inst_size, n_admm_iters], dtype=tf.float32)#1)

        var_D = list(filter( lambda x: x.name.startswith('fp32_storage/DIS'), tf.trainable_variables()))
        var_G = list(filter( lambda x: x.name.startswith('fp32_storage/PROJ'), tf.trainable_variables()))
        var_E = list(filter( lambda x: 'fp32_storage/ENCODE' in x.name, tf.trainable_variables()))

        W_D = list(filter(lambda x: x.name.endswith('W:0'), var_D))
        W_G = list(filter(lambda x: x.name.endswith('W:0'), var_G))
        W_E = list(filter(lambda x: x.name.endswith('W:0'), var_E))

        soft_pos_labels = pos_labels * one_sided_label_smooth
        neg_labels = tf.zeros([inst_size], dtype=tf.float32)
        if use_diff_admm is False:
            neg_labels_iter = neg_labels
        else:
            neg_labels_iter = tf.zeros([inst_size, n_admm_iters], dtype=tf.float32)
        loss_adv_D_pos_latent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adversarial_latentx, labels=soft_pos_labels)) # CE(D_l(E(x)), 1)
        loss_adv_D_neg_latent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adversarial_latentz, labels=neg_labels_iter)) # CE(D_l(E(v)), 0)

        loss_adv_D_pos_img = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adversarial_truex, labels=soft_pos_labels)) # CE(D(x), 1)
        loss_adv_D_neg_img = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adversarial_projx, labels=neg_labels)) # CE(D(P(x)), 0)
        loss_adv_D_neg_imgz = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adversarial_projz, labels=neg_labels_iter)) # CE(D(P(v)), 0)
        
        loss_adv_G_latent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adversarial_latentz, labels=pos_labels_iter)) # CE(D_l(E(v)), 1), term 4

        loss_adv_G_imgx = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adversarial_projx, labels=pos_labels)) # CE(D(P(x)), 1)
        loss_adv_G_imgz = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adversarial_projz, labels=pos_labels_iter)) # CE(D(P(v)), 1)

        D_var_clip_ops = list(map(lambda v: tf.assign(v, tf.clip_by_value(v, clamp_lower, clamp_upper)), W_D))
        E_var_clip_ops = list(map(lambda v: tf.assign(v, tf.clip_by_value(v, clamp_lower, clamp_upper)), W_E))
        
        loss_adv_D_latent = (lambda_ratio_tf * loss_adv_D_pos_latent + (1 - lambda_ratio_tf) * loss_adv_D_neg_latent) # CE(D_l(E(x)), 1) + CE(D_l(E(v)), 0)
        loss_adv_D_img = (loss_adv_D_pos_img + lambda_ratio_tf * loss_adv_D_neg_img + (1 - lambda_ratio_tf) * loss_adv_D_neg_imgz) * 0.5 # CE(D(x), 1) + CE(D(P(x)), 0) + CE(D(P(v)), 0)
        loss_adv_D = (lambda_latent_tf * loss_adv_D_latent + lambda_img_tf * loss_adv_D_img) # term 4 + term 5 (discriminate)
        # set up the loss for autoencoder
        loss_adv_G_img = lambda_ratio_tf * loss_adv_G_imgx + (1 - lambda_ratio_tf) * loss_adv_G_imgz # CE(D(P(x)), 1) + CE(D(P(v)), 1), term 5
        loss_adv_G = lambda_latent_tf * loss_adv_G_latent +  lambda_img_tf * loss_adv_G_img
        if use_diff_admm is False:
            loss_recon = tf.reduce_mean(tf.square(get_inst(images_tf_float32) - projection_x)) # ||x-P(x)||_2^2, term 1
            loss_recon_z = tf.reduce_mean(tf.square(get_inst(images_tf_float32) - projection_z)) # ||x-P(v)||_2^2, term 2
            loss_proj = tf.reduce_mean(tf.square(get_inst(noisy_image_tf_float32) - projection_z)) # ||v-P(v)||_2^2, term 3
            loss_G = loss_adv_G + lambda_l2_tf * (lambda_ratio_tf * loss_recon + (1-lambda_ratio_tf)*loss_proj ) + lambda_de_tf * loss_recon_z
        else:
            loss_recon = tf.reduce_mean(tf.square(tf.zeros_like(get_inst(images_tf_float32)) - u)) # ||0-admm(A,y)_u||_2^2, term 1
            loss_recon_z = tf.reduce_mean(tf.square(get_inst(images_tf_float32) - x)) # ||x-admm(A,y)_x||_2^2, term 2
            loss_proj = tf.reduce_mean(tf.square(get_inst(images_tf_float32) - z)) # ||x-admm(A,y)_z||_2^2, term 3
            loss_G = loss_proj # for now just use this term
            #loss_G = (loss_recon + loss_recon_z + loss_proj) # alternatively, can try all 3 terms, too

        if weight_decay_rate > 0:
            loss_G += weight_decay_rate * tf.reduce_mean(tf.stack( list(map(lambda x: tf.nn.l2_loss(tf.cast(x, tf.float32)), W_G)))) # \ell_2 reg for G
            loss_adv_D += weight_decay_rate * tf.reduce_mean(tf.stack( list(map(lambda x: tf.nn.l2_loss(tf.cast(x, tf.float32)), W_D)))) # \ell_2 reg for D
        
        optimizer_G = tf.train.AdamOptimizer( learning_rate=learning_rate_proj, beta1=adam_beta1_g_tf, beta2=adam_beta2_g, epsilon=adam_eps_g)
        grads_vars_G = optimizer_G.compute_gradients( loss_G, var_list=var_G )
        grads_vars_G_clipped = list(map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_G))
        train_op_G = optimizer_G.apply_gradients( grads_vars_G_clipped )
        
        optimizer_D = tf.train.AdamOptimizer( learning_rate=learning_rate_dis, beta1=adam_beta1_d_tf, beta2=adam_beta2_d, epsilon=adam_eps_d)
        grads_vars_D = optimizer_D.compute_gradients( loss_adv_D, var_list=var_D )
        grads_vars_D_clipped = list(map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_D))
        train_op_D = optimizer_D.apply_gradients( grads_vars_D_clipped )
        
        init_op_global = tf.global_variables_initializer()
        init_op_local = tf.local_variables_initializer()
    
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth=True

    sess = tf.Session(config=config_proto)

    # initialization
    sess.run(init_op_global, feed_dict={
        learning_rate_dis: learning_rate_val_dis,
        adam_beta1_d_tf: adam_beta1_d,
        learning_rate_proj: learning_rate_val_proj,
        lambda_ratio_tf: lambda_ratio,
        lambda_l2_tf: lambda_l2,
        lambda_latent_tf: lambda_latent,
        lambda_img_tf: lambda_img,
        lambda_de_tf: lambda_de,
        adam_beta1_g_tf: adam_beta1_g,
        })
    sess.run(init_op_local, feed_dict={
        learning_rate_dis: learning_rate_val_dis,
        adam_beta1_d_tf: adam_beta1_d,
        learning_rate_proj: learning_rate_val_proj,
        lambda_ratio_tf: lambda_ratio,
        lambda_l2_tf: lambda_l2,
        lambda_latent_tf: lambda_latent,
        lambda_img_tf: lambda_img,
        lambda_de_tf: lambda_de,
        adam_beta1_g_tf: adam_beta1_g,
        })


    # setup the saver
    saver = tf.train.Saver(max_to_keep=16)
    saver_epoch = tf.train.Saver(max_to_keep=100)

    # setup the image saver
    if output_img > 0:
        num_output_img = min(5, batch_size)
        output_ori_imgs_op = (images_tf[0:num_output_img] * 127.5 ) + 127.5
        if use_diff_admm is False:
            output_noisy_imgs_op = (noisy_image_tf[0:num_output_img] * 127.5 ) + 127.5
        output_project_imgs_op = (projection_z[0:num_output_img] * 127.5 ) + 127.5
        output_reconstruct_imgs_op = (projection_x[0:num_output_img] * 127.5 ) + 127.5

    if use_tensorboard > 0:
        # create a summary to monitor cost tensor
        tf.summary.scalar("loss_adv_D", loss_adv_D, collections=['dis'])
        tf.summary.scalar("loss_adv_D_pos_latent", loss_adv_D_pos_latent, collections=['dis'])
        tf.summary.scalar("loss_adv_D_neg_latent", loss_adv_D_neg_latent, collections=['dis'])
        tf.summary.scalar("loss_adv_D_latent", loss_adv_D_latent, collections=['dis'])
        tf.summary.scalar("loss_adv_D_pos_img", loss_adv_D_pos_img, collections=['dis'])
        tf.summary.scalar("loss_adv_D_neg_img", loss_adv_D_neg_img, collections=['dis'])
        tf.summary.scalar("loss_adv_D_neg_imgz", loss_adv_D_neg_imgz, collections=['dis'])
        tf.summary.scalar("loss_adv_D_img", loss_adv_D_img, collections=['dis'])

        tf.summary.scalar("loss_G", loss_G, collections=['proj'])
        tf.summary.scalar("loss_adv_G_latent", loss_adv_G_latent, collections=['proj'])
        tf.summary.scalar("loss_adv_G_imgx", loss_adv_G_imgx, collections=['proj'])
        tf.summary.scalar("loss_recon_z", loss_recon_z, collections=['proj'])
        tf.summary.scalar("loss_recon", loss_recon, collections=['proj'])
        tf.summary.scalar("loss_proj", loss_proj, collections=['proj'])
        tf.summary.scalar("lambda_ratio", lambda_ratio_tf, collections=['proj'])
        tf.summary.scalar("lambda_l2", lambda_l2_tf, collections=['proj'])
        tf.summary.scalar("lambda_latent", lambda_latent_tf, collections=['proj'])
        tf.summary.scalar("lambda_img", lambda_img_tf, collections=['proj'])
        tf.summary.scalar("lambda_de", lambda_de_tf, collections=['proj'])
        tf.summary.scalar("adam_beta1_g", adam_beta1_g_tf, collections=['proj'])
        tf.summary.scalar("adam_beta1_d", adam_beta1_d_tf, collections=['proj'])
        tf.summary.scalar("learning_rate_proj", learning_rate_proj, collections=['proj'])
        tf.summary.scalar("learning_rate_dis", learning_rate_dis, collections=['proj'])
        tf.summary.image("original_image", images_tf, max_outputs=5, collections=['proj'])
        if use_diff_admm is False:
            tf.summary.image("noisy_image", noisy_image_tf, max_outputs=5, collections=['proj'])
            tf.summary.image("projected_z", projection_z, max_outputs=5, collections=['proj'])
        tf.summary.image("reconstructed_x", projection_x, max_outputs=5, collections=['proj'])

        # merge all summaries into a single op
        summary_G = tf.summary.merge_all(key='proj')
        summary_D = tf.summary.merge_all(key='dis')

    print('reload previously trained model')
    if use_pretrain == True:
        print('reloading %s...' % pretrained_model_file)
        saver.restore( sess, pretrained_model_file )

    if use_tensorboard > 0:
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        print("Run the command line:\n --> tensorboard --logdir=%s\n" %  logs_base)
        print("Then open http://0.0.0.0:6006/ into your web browser")


    # continue the iteration number
    if use_pretrain == True:
        iters = pretrained_iter + 1
    else:
        iters = 0

    start_epoch = iters // (total_train // batch_size)


    print('start training')
    start_time = timeit.default_timer()

    iters_in_epoch = total_train // batch_size
    epoch = 0

    loss_G_avg = SmoothStream(window_size=100)
    loss_recon_avg = SmoothStream(window_size=100)
    loss_recon_z_avg = SmoothStream(window_size=100)
    loss_proj_avg = SmoothStream(window_size=100)
    loss_adv_G_latent_avg = SmoothStream(window_size=100)
    loss_adv_G_img_avg = SmoothStream(window_size=100)
    loss_dis_avg = SmoothStream(window_size=100)
    loss_dis_latent_avg = SmoothStream(window_size=100)
    loss_dis_img_avg = SmoothStream(window_size=100)

    update_D_left = 0
    update_G_left = Gperiod

    loss_G_val = 0
    loss_recon_val = 0
    loss_recon_z_val = 0
    loss_proj_val = 0
    loss_adv_G_latent_val = 0
    loss_adv_G_img_val = 0
    loss_D_val = 0
    loss_D_latent_val = 0
    loss_D_img_val = 0


    print('alternative training starts....')

    while True:
        if use_diff_admm is False:
            inst_batch, inst_noisy_batch = train_queue.get()
            batch = np_combine_batch(inst_batch, ref_batch)
            noisy_batch = np_combine_batch(inst_noisy_batch, ref_noisy_batch)
        else:
            batch, ATy_batch, z0_batch, Qindices, Qids, Qweights, Qshape, rho = train_queue.get()
        
        # adjust learning rate
        learning_rate_val_proj_current = 2e-1 / lambda_de
        learning_rate_val_proj_current = min(learning_rate_val_proj, learning_rate_val_proj_current)


        if update_G_left > 0 and update_D_left <= 0:

            sys.stdout.write('G: ')

            # update G
            if use_diff_admm is False:
                _, loss_G_val, loss_proj_val, loss_recon_val, loss_recon_z_val, loss_adv_G_latent_val, loss_adv_G_img_val, _ = sess.run(
                    [train_op_G, loss_G, loss_proj, loss_recon, loss_recon_z, loss_adv_G_latent, loss_adv_G_img, updates],
                    feed_dict={
                        images_tf: batch,
                        noisy_image_tf: noisy_batch,
                        learning_rate_dis: learning_rate_val_dis,
                        adam_beta1_d_tf: adam_beta1_d,
                        learning_rate_proj: learning_rate_val_proj_current,
                        lambda_ratio_tf: lambda_ratio,
                        lambda_l2_tf: lambda_l2,
                        lambda_latent_tf: lambda_latent,
                        lambda_img_tf: lambda_img,
                        lambda_de_tf: lambda_de,
                        adam_beta1_g_tf: adam_beta1_g,
                    })
            else:
                _, loss_G_val, loss_proj_val, loss_recon_val, loss_recon_z_val, loss_adv_G_latent_val, loss_adv_G_img_val, _ = sess.run(
                    [train_op_G, loss_G, loss_proj, loss_recon, loss_recon_z, loss_adv_G_latent, loss_adv_G_img, updates],
                    feed_dict={
                        images_tf: batch,
                        ATy_tf: ATy_batch,
                        z0_tf: z0_batch,
                        Qindices_tf: Qindices,
                        Qids_tf: Qids,
                        Qweights_tf: Qweights,
                        Qshape_tf: Qshape,
                        rho_tf: rho,
                        learning_rate_dis: learning_rate_val_dis,
                        adam_beta1_d_tf: adam_beta1_d,
                        learning_rate_proj: learning_rate_val_proj_current,
                        lambda_ratio_tf: lambda_ratio,
                        lambda_l2_tf: lambda_l2,
                        lambda_latent_tf: lambda_latent,
                        lambda_img_tf: lambda_img,
                        lambda_de_tf: lambda_de,
                        adam_beta1_g_tf: adam_beta1_g,
                    })
                        
            update_G_left -= 1
            loss_G_avg.insert(loss_G_val)
            loss_recon_avg.insert(loss_recon_val)
            loss_recon_z_avg.insert(loss_recon_z_val)
            loss_proj_avg.insert(loss_proj_val)
            loss_adv_G_latent_avg.insert(loss_adv_G_latent_val)
            loss_adv_G_img_avg.insert(loss_adv_G_img_val)
            if update_G_left <= 0:
                update_D_left = Dperiod

        if update_G_left <= 0 and update_D_left > 0:

            sys.stdout.write('D: ')

            # update D
            if use_diff_admm is False:
                _, loss_D_val, loss_D_latent_val, loss_D_img_val, _ = sess.run(
                    [train_op_D, loss_adv_D, loss_adv_D_latent, loss_adv_D_img, updates],
                    feed_dict={
                        images_tf: batch,
                        noisy_image_tf: noisy_batch,
                        learning_rate_dis: learning_rate_val_dis,
                        adam_beta1_d_tf: adam_beta1_d,
                        learning_rate_proj: learning_rate_val_proj_current,
                        lambda_ratio_tf: lambda_ratio,
                        lambda_l2_tf: lambda_l2,
                        lambda_latent_tf: lambda_latent,
                        lambda_img_tf: lambda_img,
                        lambda_de_tf: lambda_de,
                        adam_beta1_g_tf: adam_beta1_g,
                    })
            else:
                _, loss_D_val, loss_D_latent_val, loss_D_img_val, _ = sess.run(
                    [train_op_D, loss_adv_D, loss_adv_D_latent, loss_adv_D_img, updates],
                    feed_dict={
                        images_tf: batch,
                        ATy_tf: ATy_batch,
                        z0_tf: z0_batch,
                        Qindices_tf: Qindices,
                        Qids_tf: Qids,
                        Qweights_tf: Qweights,
                        Qshape_tf: Qshape,
                        rho_tf: rho,
                        learning_rate_dis: learning_rate_val_dis,
                        adam_beta1_d_tf: adam_beta1_d,
                        learning_rate_proj: learning_rate_val_proj_current,
                        lambda_ratio_tf: lambda_ratio,
                        lambda_l2_tf: lambda_l2,
                        lambda_latent_tf: lambda_latent,
                        lambda_img_tf: lambda_img,
                        lambda_de_tf: lambda_de,
                        adam_beta1_g_tf: adam_beta1_g,
                    })

            if clamp_weight > 0:
                # clip the variables of the discriminator
                _,_ = sess.run([D_var_clip_ops, E_var_clip_ops])

            update_D_left -= 1
            loss_dis_avg.insert(loss_D_val)
            loss_dis_latent_avg.insert(loss_D_latent_val)
            loss_dis_img_avg.insert(loss_D_img_val)

            if update_D_left <= 0:
                update_G_left = Gperiod

        if update_G_left <= 0 and update_D_left <= 0:
            update_D_left = 0
            update_G_left = Gperiod


        print("Iter %d (%.2fm): l_G=%.3e l_rec=%.3e l_rec_z=%.3e l_proj=%.3e l_G_lat:%.3e l_G_img:%.3e l_D=%.3e l_D_lat=%.3e l_D_img=%.3e lrp=%.3e lrd=%.3e qsize=%d" % (
                iters, (timeit.default_timer()-start_time)/60., 
                loss_G_avg.get_moving_avg(),
                loss_recon_avg.get_moving_avg(), loss_recon_z_avg.get_moving_avg(), loss_proj_avg.get_moving_avg(), 
                loss_adv_G_latent_avg.get_moving_avg(), loss_adv_G_img_avg.get_moving_avg(),
                loss_dis_avg.get_moving_avg(),
                loss_dis_latent_avg.get_moving_avg(),
                loss_dis_img_avg.get_moving_avg(),
                learning_rate_val_proj, learning_rate_val_dis, train_queue.qsize()))


        if (iters + 1) % 2000 == 0:
            saver.save(sess, model_path + '/model_iter', global_step=iters)


        # output to tensorboard
        if use_tensorboard >0 and (iters % tensorboard_period == 0):
            if use_diff_admm is False:
                summary_d_vals, summary_g_vals = sess.run(
                    [summary_D, summary_G],
                    feed_dict={
                        images_tf: batch,
                        noisy_image_tf: noisy_batch,
                        learning_rate_dis: learning_rate_val_dis,
                        learning_rate_proj: learning_rate_val_proj_current,
                        lambda_ratio_tf: lambda_ratio,
                        lambda_l2_tf: lambda_l2,
                        lambda_latent_tf: lambda_latent,
                        lambda_img_tf: lambda_img,
                        lambda_de_tf: lambda_de,
                        adam_beta1_g_tf: adam_beta1_g,
                        adam_beta1_d_tf: adam_beta1_d
                    })
            else:
                summary_d_vals, summary_g_vals = sess.run(
                    [summary_D, summary_G],
                    feed_dict={
                        images_tf: batch,
                        ATy_tf: ATy_batch,
                        z0_tf: z0_batch,
                        Qindices_tf: Qindices,
                        Qids_tf: Qids,
                        Qweights_tf: Qweights,
                        Qshape_tf: Qshape,
                        rho_tf: rho,
                        learning_rate_dis: learning_rate_val_dis,
                        learning_rate_proj: learning_rate_val_proj_current,
                        lambda_ratio_tf: lambda_ratio,
                        lambda_l2_tf: lambda_l2,
                        lambda_latent_tf: lambda_latent,
                        lambda_img_tf: lambda_img,
                        lambda_de_tf: lambda_de,
                        adam_beta1_g_tf: adam_beta1_g,
                        adam_beta1_d_tf: adam_beta1_d
                    })

            summary_writer.add_summary(summary_g_vals, iters)
            summary_writer.add_summary(summary_d_vals, iters)

        # save some images
        if output_img > 0 and (iters + 1) % output_img_period == 0:
            if use_diff_admm is False:
                output_ori_img_val, output_noisy_img_val, output_project_img_val, output_reconstruct_imgs_val = sess.run(
                    [output_ori_imgs_op, output_noisy_imgs_op, output_project_imgs_op, output_reconstruct_imgs_op],
                    feed_dict={
                        images_tf: batch,
                        noisy_image_tf: noisy_batch,
                    }
                )
            else:
                output_ori_img_val, output_noisy_img_val, output_project_img_val, output_reconstruct_imgs_val = sess.run(
                    [output_ori_imgs_op, output_noisy_imgs_op, output_project_imgs_op, output_reconstruct_imgs_op],
                    feed_dict={
                        images_tf: batch,
                        ATy_tf: ATy_batch,
                        z0_tf: z0_batch,
                        Qindices_tf: Qindices,
                        Qids_tf: Qids,
                        Qweights_tf: Qweights,
                        Qshape_tf: Qshape,
                        rho_tf: rho
                    }
                )
            output_folder = '%s/iter_%d' %(img_path, iters)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            for i in range(output_ori_img_val.shape[0]):
                filename = '%s/%d_ori.jpg' % (output_folder, i)
                imsave(filename, output_ori_img_val[i].astype('uint8'))
                filename = '%s/%d_noisy.jpg' % (output_folder, i)
                imsave(filename, output_noisy_img_val[i].astype('uint8'))
                filename = '%s/%d_proj.jpg' % (output_folder, i)
                imsave(filename, output_project_img_val[i].astype('uint8'))
                filename = '%s/%d_recon.jpg' % (output_folder, i)
                imsave(filename, output_reconstruct_imgs_val[i].astype('uint8'))

        iters += 1

        lambda_de *= de_decay_rate


        if iters % iters_in_epoch == 0:
            epoch += 1
            saver_epoch.save(sess, epoch_path + '/model_epoch', global_step=epoch)
            learning_rate_val_dis *= 0.95
            learning_rate_val_proj *= 0.95
            if epoch > n_epochs:
                break

        # recreate new train_proc (force garbage colection)
        if iters % 2000 == 0:
            terminate_train_procs(train_procs)
            del train_procs
            del train_queue
            train_queue = Queue(num_prepare+1)
            train_procs = []
            create_train_procs(trainset, train_queue, n_thread, num_prepare, train_procs, use_diff_admm, dtype)
            
    sess.close()
