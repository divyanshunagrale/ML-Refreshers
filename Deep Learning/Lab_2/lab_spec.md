# Lab 2: Dual-Mode Image Segmentation (U-Net)

## 1.The Lore & Objective

An autonomous drone delivery company needs to parse high-resolution camera feeds to separate "safe landing zones" (grass/flat ground) from "hazard zones" (water/buildings) at a pixel level.

Instead of classifying an _entire image_ as "safe" or "unsafe," you must assign a label to **every single pixel**.You will construct a U-Net model twice:

- **Phase A:** Build a completely custom deep CNN encoder from scratch to master spatial tensor arithmetic.

- **Phase B:** Use Transfer Learning to replace your custom encoder with a pre-trained feature extractor (`MobileNetV2`), freeze its weights, and connect it to your decoder using the **Keras Functional API**.

## 2.The Comprehensive U-Net Reading Room

### What is Semantic Segmentation?

Unlike standard classification (which answers _"What is in this image?"_), semantic segmentation answers _"Where is every object at a pixel level?"_.Your input is an image of shape `(Height, Width, 3)`, and your target ground-truth is a grayscale **mask** of shape `(Height, Width, 1)`, where each pixel value represents an integer class ID or probability.

### Structural Anatomy of a U-Net

Originally created for medical image processing, the U-Net is shaped exactly like a "U." It consists of:

1.**The Contracting Path (Encoder):** A standard CNN architecture. It applies consecutive Convolutions and Pooling layers.This halves spatial dimensions (`Height`, `Width`) while doubling depth (`Channels`).It captures _what_ features are present but loses _where_ they were located.

2.**The Bottleneck:** The lowest layer of the network connecting the encoder and decoder.

3.**The Expansive Path (Decoder):** It uses up-sampling or transpose convolutions to scale spatial dimensions back up to the original size.

4.**Skip Connections (The Secret Sauce):** Because down-sampling loses precise spatial boundaries, U-Net takes high-resolution feature maps from the encoder and **concatenates** them directly onto the corresponding up-sampled maps in the decoder.This preserves edge fidelity.

### Spatial Dimension Math (Essential for Interviews)

When a feature map passes through a standard 2D convolution, the output width ($W_{out}$) and height ($H_{out}$) are governed by:

$$W_{out} = \lfloor \frac{W_{in} - K + 2P}{S} \rfloor + 1$$

Where $K$ is Kernel Size, $P$ is Padding, and $S$ is Stride.

- If you use `padding='same'` in Keras, the framework automatically pads the borders so that if $S=1$, $W_{out} = W_{in}$.

- A `MaxPooling2D` layer with a pool size of `(2,2)` simply cuts height and width exactly in half.

## 3.Conceptual Code Guidance (No Spoilers)

To build U-Net, you cannot use `tf.keras.Sequential`.You _must_ use the **Keras Functional API**, because the data paths split and merge via skip connections.

### Understanding the Keras Functional API

In the Functional API, layers behave like mathematical functions.You pass the previous tensor directly into the new layer as a call argument:

``` Plaintext
input_tensor = Defining an input placeholder shape
x = Call Conv2D layer parameters (pass input_tensor here)
x = Call MaxPooling2D layer parameters (pass x here)

```

### How to construct Phase A (Encoder from scratch)

Your encoder needs to step down 3 times. At each step, apply two consecutive `Conv2D` layers (with `padding='same'` and `activation='relu'`).Save the output of the second convolution—this will be your skip connection! Then, apply `MaxPooling2D` to reduce dimensions before moving to the next step.

``` Plaintext
Step 1: Input (e.g., 128x128x3) -> Conv -> Conv (Save this as Skip_1) -> MaxPool (64x64)
Step 2: MaxPool Output -> Conv -> Conv (Save this as Skip_2) -> MaxPool (32x32)

```

### How to construct the Decoder and Skip Concatenations

To bring your spatial scale back up, you will use `UpSampling2D(size=(2,2))` or `Conv2DTranspose`.

Once up-sampled, you must merge it with your saved skip connection using `tf.keras.layers.Concatenate()`:

``` Plaintext
Upsampled_Tensor = UpSample Layer()(Previous_Tensor)
Merged_Tensor = Concatenate Layer()([Upsampled_Tensor, Saved_Skip_Tensor])
x = Conv2D()(Merged_Tensor)

```

### How to handle Phase B (Transfer Learning Backbone)

Instead of manually typing Step 1 and Step 2, you load a pre-trained model:

``` Plaintext
backbone = Call tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_tensor=...)

```

To get your skip connections, you query specific internal layers of `backbone` by their string names or indices and pull their `.output` properties, feeding those into your custom decoder blocks!

## 4.Step-by-Step Lab Guide

### Milestone 1: Data Preparation

**The Oxford-IIIT Pet Dataset.**

Instead of dealing with full pet breed classification (which is too complex for a fast refresher), you will use its built-in **Trimap Segmentation Masks**.Every image contains a pet, and the matching target mask marks out three exact pixel classes:

- **Class 1:** The interior of the pet.

- **Class 2:** The background.

- **Class 3:** The boundary/outline of the pet.

### Data Preparation Guide

Because this dataset is fully integrated into the `tensorflow_datasets` (TFDS) ecosystem, you don't need to manually scrape images or write complex file-parsing directories.Here is exactly how to set up your pipeline.

#### 1.Importing the Target Split

You can pull down the dataset already partitioned into training and validation groups directly through the TFDS module:

- Use `tfds.load('oxford_iiit_pet:3.*.*', with_info=True)` to fetch both the raw pipeline dictionaries and an metadata information object.

- This dataset provides an image array under the dictionary key `'image'` and its spatial mask array under `'segmentation_mask'`.

#### 2.Resolving the Target Mask Coordinates (The Trap)

The default segmentation masks in this dataset are encoded with pixel integer values of **`{1, 2, 3}`**.

- Standard multi-class classification losses in Keras (like `SparseCategoricalCrossentropy`) require zero-based indices starting at **`0`**.

- **Your Task:** In your image transformation processing block, you _must_ subtract `1` from the mask values so that the integers transform cleanly into **`{0, 1, 2}`**.If you forget this step, your loss evaluation will attempt to look up index 3, causing an out-of-bounds classification failure.

#### 3.Spatial and Numerical Rescaling

- **Images:** The raw source photographs vary in shape and resolution.You will scale their dimensions down to a uniform frame size of `(128, 128, 3)` using regular bilinear resizing, and normalize their pixel values by dividing by `255.0` to pull the range into `[0.0, 1.0]`.

- **Masks:** When resizing the target masks down to `(128, 128, 1)`, **do not use standard bilinear resizing.** Bilinear scaling introduces interpolated float values (e.g., trying to average a pixel between background class 0 and boundary class 2 outputs 1.0).You must use **Nearest-Neighbor interpolation** (`method=tf.image.ResizeMethod.NEAREST_NEIGHBOR`) to preserve strict integer boundaries.

### Step-by-Step Data Pipeline Pseudocode Strategy

When assembling this in your project notebook or script, map out your `tf.data` workflow using the following logic sequence:

1. **Write a helper mapping function** (`load_images_and_masks(datapoint)`):

    - Extract the raw image, convert it to `tf.float32`, scale it to `(128, 128)`, and divide by `255.0`.

    - Extract the raw segmentation mask, scale it to `(128, 128)` using nearest-neighbor scaling, and subtract `1`.

    - Return the processed tuple: `(rescaled_image, zero_indexed_mask)`.

2. **Construct the optimized batching pipeline**:

    - Apply `.map()` to pass your raw datasets through your helper mapping function.

    - Chain on `.shuffle(buffer_size=1000)` to the training data split to ensure randomized batch propagation.

    - Attach `.batch(batch_size=32)` or `.batch(64)` to assemble your tensor groups.

    - Conclude with `.prefetch(buffer_size=tf.data.AUTOTUNE)`.This ensures that while your GPU is actively computing gradient transformations on batch $N$, the CPU is already preprocessing batch $N+1$ in memory, completely eliminating pipeline execution bottlenecks.

### Milestone 2: Build the Scratch U-Net (Phase A)

1.Initialize an entry `Input` node.

2.Code an encoder that down-samples twice, keeping track of your tensor variables for the skip links.

3.Code a bottleneck layer.

4.Code a decoder that up-samples twice, utilizing the `Concatenate` layer at each matching dimension.

5.Set your final layer to a `Conv2D` with 1 filter and a `'sigmoid'` activation (for binary background vs foreground parsing).Create the final model passing inputs and outputs.

### Milestone 3: Swap in the Transfer Learning Engine (Phase B)

1.Set up a fresh model construction routine.

2.Load `MobileNetV2` with frozen weights (`layer.trainable = False`).

3.Extract specific feature activation layers to serve as your new encoder skip layers.

4.Route those outputs directly into your existing Phase A decoder framework.

## 5.Interview Defense Targets

- _Why use a `Concatenate` layer instead of an `Add` layer when combining encoder and decoder steps? What is the architectural difference?_

- _If your target masks contain a massive class imbalance (e.g., 98% background pixels and only 2% target object pixels), why will standard Pixel-wise Binary Cross-Entropy completely fail, and how does Dice Loss resolve this?_

To get an intuitive visual walkthrough of how these blocks connect before you begin scripting, check out this video on the [U-Net architecture clearly explained](https://www.youtube.com/watch?v=oxcgx75k6yU).It breaks down the spatial mechanics and layer interactions using clean animations that help demystify the Keras Functional API layout.
