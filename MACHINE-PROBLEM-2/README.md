# Hands-On Exploration:
* **Lab Session 1: Image Transformations**
  * **Scaling and Rotation:** Learn how to apply scaling and rotation transformations to images
using OpenCV.
  * **Implementation:** Practice these transformations on sample images provided in the lab.
  ```python
  image = cv2.imread('/content/Untitled.jpg')
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  scale_percent = 50
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  dim = (width, height)
  resized_image = cv2.resize(image, dim)
  
  angle = 45
  center = (image.shape[1] // 2, image.shape[0] // 2)
  rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
  rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
  
  fig, axes = plt.subplots(1, 3, figsize=(10, 5))
  
  axes[0].imshow(image)
  axes[0].set_title("Original Image")
  
  axes[1].imshow(resized_image)
  axes[1].set_title("Scaled Image")
  
  axes[2].imshow(rotated_image)
  axes[2].set_title("Rotated Image")
  
  
  plt.tight_layout()
  plt.show()
  ```
  ![Untitled](assets\outputs\output.png)
  ![Untitled](assets\outputs\output1.png)

* **Lab Session 2: Filtering Techniques**
  * **Blurring and Edge Detection:** Explore how to apply blurring filters and edge detection algorithms to images using OpenCV.
  * **Implementation:** Apply these filters to sample images to understand their effects.
```python
Gaussian = cv2.GaussianBlur(image, (5, 5), 0)
median = cv2.medianBlur(image, 5)
bilateral = cv2.bilateralFilter(image, 9, 75, 75)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0][0].imshow(image)
axes[0][0].set_title("Original Image")

axes[0][1].imshow(Gaussian)
axes[0][1].set_title("Gaussian Blur")

axes[1][0].imshow(median)
axes[1][0].set_title("Median Blur")

axes[1][1].imshow(bilateral)
axes[1][1].set_title("Bilateral Blur")

plt.tight_layout()
plt.show()
```

  ![Untitled](assets\outputs\3.png)

```python

orig_edges = cv2.Canny(image, 100, 200)
gaussian_edges = cv2.Canny(Gaussian, 100, 200)
median_edges = cv2.Canny(median, 100, 200)
bilateral_edges = cv2.Canny(bilateral, 100, 200)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0][0].imshow(orig_edges,cmap='gray')
axes[0][0].set_title("Original Image")

axes[0][1].imshow(gaussian_edges,cmap='gray')
axes[0][1].set_title("Gaussian Blur")

axes[1][0].imshow(median_edges,cmap='gray')
axes[1][0].set_title("Median Blur")

axes[1][1].imshow(bilateral_edges,cmap='gray')
axes[1][1].set_title("Bilateral Blur")

plt.tight_layout()
plt.show()
```
  ![Untitled](assets\outputs\4.png)

# **Problem-Solving Session:**
* **Common Image Processing Tasks:**
  * Engage in a problem-solving session focused on common challenges encountered in image processing tasks.
  * **Scenario-Based Problems:** Solve scenarios where you must choose and apply appropriate image processing techniques.

Images captured in real world are subjected to noise due to environment, signal instability, camera sensor issues, poor lighting conditions, electrical loss etc. For further processing these images and to interpret results it is essential to have images with noise as low as possible.

Image Denoising is a critical process in digital image processing, aiming to enhance the visual quality of images by reducing noise. It is challenging topic of active research, involved in understanding the type of noise in the image and thereby apply a denoising method that can reduce the noise and provide more accurate representation of the original image.

Images serve as powerful tools for communication, analysis, and interpretation across multiple domains. However, their fidelity can be compromised due to various types of noise, affecting their quality and interpretability. Let’s delve into the distinct types of noise that commonly plague images:

1. **Gaussian Noise**

  Gaussian noise emerges from random variations following a Gaussian distribution. It appears as a subtle, uniform noise that can be introduced during the image acquisition process. Typically, it blurs edges, reduces image clarity, and stems from sources like electronic sensors or transmission errors.

  ```python
  def add_gaussian_noise(image):
    mean = 0
    std_dev = 25
    h, w, c = image.shape
    gaussian = np.random.normal(mean, std_dev, (h, w, c))
    noisy_image = np.clip(image + gaussian, 0, 255).astype(np.uint8)
    return noisy_image
```

2. **Salt-and-Pepper Noise**

  This type of noise manifests as isolated white and black pixels scattered across an image, akin to grains of salt and pepper. These random pixels are often a result of errors in data transmission or storage. Salt-and-pepper noise can obscure fine details and significantly impact visual interpretation.

```python
  def add_salt_and_pepper_noise(image, amount=0.05):
    noisy_image = np.copy(image)
    num_salt = np.ceil(amount * image.size * 0.5)
    salt_coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape]
    noisy_image[salt_coords] = 255
  
    num_pepper = np.ceil(amount * image.size * 0.5)
    pepper_coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape]
    noisy_image[pepper_coords] = 0
    return noisy_image
```

3. **Speckle noise**

  Speckle noise, prevalent in images acquired through ultrasound or synthetic aperture radar, causes random brightness or darkness variations. It blurs fine details, altering pixel intensities and presenting challenges for image analysis and interpretation.

  ```python
  
  def add_speckle_noise(image):
    h, w, c = image.shape
    speckle = np.random.randn(h, w, c)
    noisy_image = np.clip(image + image * speckle * 0.1, 0, 255).astype(np.uint8)
    return noisy_image
```

4. **Poisson Noise**

  Arising from the Poisson distribution, this noise is common in low-light photography or astronomical imaging. It appears as grainy artifacts and reduces image contrast and clarity in conditions with minimal light, affecting overall image quality.
  ```python
  def add_poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy_image = np.random.poisson(image * vals) / float(vals)
    return noisy_image.astype(np.uint8)
```
5. **Periodic or Banding Noise**

  Periodic noise presents as regular patterns or bands across an image, often due to interference or sensor issues. It distorts fine details, altering the image’s appearance and posing challenges for accurate interpretation.
  ```python
  def add_periodic_noise(image, frequency=16):
    h, w, c = image.shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    noise = np.sin(2 * np.pi * frequency * Y / h) * 128 + 128
    noisy_image = np.clip(image + noise[..., np.newaxis], 0, 255).astype(np.uint8)
    return noisy_image
```

```python
image = cv2.imread('Untitled.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gaussian_noise = add_gaussian_noise(image)
salt_and_pepper_noise = add_salt_and_pepper_noise(image)
speckle_noise = add_speckle_noise(image)
poisson_noise = add_poisson_noise(image)
periodic_noise = add_periodic_noise(image)

fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes[0,0].set_title("Original Image")
axes[0,0].imshow(image)

axes[0,1].set_title("Gausian Noise")
axes[0,1].imshow(gaussian_noise)

axes[0,2].set_title("Salt and Pepper Noise")
axes[0,2].imshow(salt_and_pepper_noise)

axes[1,0].set_title("Speckle Noise")
axes[1,0].imshow(speckle_noise)

axes[1,1].set_title("Poisson Noise")
axes[1,1].imshow(poisson_noise)

axes[1,2].set_title("periodic_noise")
axes[1,2].imshow(periodic_noise)
```

Understanding these various types of noise is crucial for image processing and analysis. Each type exhibits distinct characteristics that impact image fidelity differently, motivating the development of robust denoising techniques.

By comprehensively recognizing these noise variations, researchers and practitioners in image processing aim to devise effective denoising strategies. Advanced techniques, including machine learning-driven approaches like autoencoders, strive to mitigate these noise influences, restoring image quality and enabling accurate analysis across diverse applications.


## Classical Approach for Image Denoising

**Spatial domain filtering:** Spatial domain filtering involves manipulating an image’s pixel values directly in the spatial domain (the image itself) to enhance or modify its characteristics. Different filter types are used in spatial domain filtering to achieve various effects:

1. **Smoothing Filters:**
  * **Mean Filter:** Replaces each pixel’s value with the average of its neighboring pixels. It reduces noise but may blur edges and fine details.

  * **Gaussian Filter:** Assigns weights to neighboring pixels based on a Gaussian distribution. It smoothens the image while preserving edges better than mean filtering.
  
2. **Sharpening Filters:**

  * **Laplacian Filter:** Enhances edges by highlighting sudden intensity changes. It emphasizes edges but also amplifies noise.

  * **High-pass Filter:** Emphasizes high-frequency components, making edges and details more pronounced. It can enhance images but may also amplify noise.
    
3. **Edge Detection Filters:**
  * **Sobel and Prewitt Filters:** Identify edges by calculating gradients in horizontal and vertical directions. They highlight edges with different orientations.

  * **Canny Edge Detector:** More advanced, it uses multiple steps to detect edges accurately by suppressing noise and finding local maxima in gradients.
    
4. **Other Filters:**
  * **Median Filter:** Replaces a pixel’s value with the median of neighboring pixels. Effective in removing salt-and-pepper noise while preserving edges.

  * **Bilateral Filter:** Retains edges while reducing noise. It smoothens images based on both spatial and intensity differences.
    
  Spatial Domain Filtering Process:

```python
  filtered_mean = cv2.blur(gaussian_noise, (5, 5))  # Mean filter (5x5 kernel)
  filtered_gaussian = cv2.GaussianBlur(gaussian_noise, (5, 5), 0)  # Gaussian filter (5x5 kernel)
  filtered_median = cv2.medianBlur(gaussian_noise, 5)  # Median filter (5x5 kernel)
  filtered_bilateral = cv2.bilateralFilter(gaussian_noise,9,75,75)
  laplacian = cv2.Laplacian(gaussian_noise, cv2.CV_8U)  # Laplacian filter
  sobel_x = cv2.Sobel(gaussian_noise, cv2.CV_8U, 1, 0, ksize=5)  # Sobel X filter (5x5 kernel)
  sobel_y = cv2.Sobel(gaussian_noise, cv2.CV_8U, 0, 1, ksize=5)  # Sobel Y filter (5x5 kernel)
  canny = cv2.Canny(gaussian_noise, 100, 200)  # Canny edge detection
  
  
  filtered_images = [filtered_mean, filtered_gaussian, filtered_median, filtered_bilateral, laplacian, sobel_x, sobel_y, canny]
  titles = ['Mean Filter', 'Gaussian Filter', 'Median Filter', 'Bilateral Filter',
            'Laplacian Filter', 'Sobel X', 'Sobel Y', 'Canny Edge Detection']
  
  fig, axes = plt.subplots(2, 4, figsize=(15, 8))
  axes = axes.ravel()
  
  for i in range(len(filtered_images)):
      if i == 7:  # Canny produces a binary image, use gray colormap
          axes[i].imshow(filtered_images[i], cmap='gray')
      else:
          axes[i].imshow(filtered_images[i])
      axes[i].set_title(titles[i])
      axes[i].axis("off")
  
  plt.tight_layout()
  plt.show()

```

# **Assignment:**
* **Implementing Image Transformations and Filtering:**
  * Choose a set of images and apply the techniques you've learned, including scaling,
rotation, blurring, and edge detection.
  * **Documentation:** Document the steps taken, and the results achieved in a report.
```python
image = cv2.imread('/content/Untitled.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

scale_percent = 50
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(image, dim)

angle = 45
center = (image.shape[1] // 2, image.shape[0] // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

# Gaussian Blur
Gaussian = cv2.GaussianBlur(image, (5, 5), 0)
# Median Blur
median = cv2.medianBlur(image, 5)
# Bilateral Blur
bilateral = cv2.bilateralFilter(image, 9, 75, 75)

fig, axes = plt.subplots(2, 3, figsize=(10, 5))

axes[0][0].imshow(image)
axes[0][0].set_title("Original Image")

axes[0][1].imshow(resized_image)
axes[0][1].set_title("Scaled Image")

axes[0][2].imshow(rotated_image)
axes[0][2].set_title("Rotated Image")

axes[1][0].imshow(Gaussian)
axes[1][0].set_title("Gaussian Filter")

axes[1][1].imshow(median)
axes[1][1].set_title("Median Filter")

axes[1][2].imshow(bilateral)
axes[1][2].set_title("Bilateral Filter")




plt.tight_layout()
plt.show()
```

Image Loading and Conversion Load the Image, During this process we will convert image from BGR to RGB otherwise output might not be displayed properly.

Resizing: Produce a scaled-down (half size) version of the image

Rotate: Rotate the image 45 degrees to further illustrate how orientation is included into the mix.

Filtering:

  Gaussian Blur: Blurs the image slightly and reduces granularity.
  
  Median Blur: This filter reduces salt-and-pepper noise and preserves edges.
  
  Bilateral Filter -Reduces noise and smoothen the image while maintaining edges.
  
  Image without filter : The original base image
  
  Scaled Image: A smaller version of image being resized
  
  Rotated Image: 45 degrees rotated from original image
  
  Gaussian Filter: Image after a Gaussian blur has been applied.
  
  Median Filter: Image after applying median blur to reduce noise.
  
  Bilateral Filter: Image after reducing Noise without losing edges using bilateral filter.



