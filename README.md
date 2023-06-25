# FFgan
<i>A cutting-edge program that leverages the power of Generative Adversarial Networks (GANs) to generate synthetic faces.</i>

<h1>Description</h1>
<p>FFgan is an innovative program that harnesses the power of Generative Adversarial Networks (GANs) to generate synthetic faces. GANs consist of two neural networks, a generator, and a discriminator, engaged in a competitive learning process. The generator network learns to produce realistic synthetic faces, while the discriminator network learns to distinguish between real and synthetic faces.</p>

<p>By leveraging the advanced capabilities of the Dlib library for image processing and incorporating a specific DCGAN model for color image generation, FFgan offers a powerful and versatile platform for face generation. It enables users to create lifelike synthetic faces with exceptional realism and diversity.</p>

<p>This technology has significant implications for both research and model production. In research, FFgan provides a valuable tool for studying facial characteristics, exploring variations in facial expressions, and analyzing demographic trends without relying on real-world data. It offers researchers the ability to generate large datasets with controlled variables for training and evaluating facial recognition systems, expression analysis algorithms, and more.</p>

<p>FFgan opens up new possibilities for designing and iterating on novel face models. It allows designers and developers to experiment with different facial features, expressions, and attributes, accelerating the development of innovative applications in various fields, including computer graphics, entertainment, virtual reality, and character design.</p>

<p>An additional advantage of FFgan is its contribution to the protection of personal data. As the generation process relies solely on synthetic data, there is no need to use real faces or store large-scale personal datasets. This mitigates privacy concerns and reduces the risk of data breaches, making it a privacy-friendly solution for face-related research and applications.</p>

<h2>Features</h2>
<ul>
  <li>GAN-based face generation.</li>
  <li>Integration of Dlib library for image processing and IA model (also the Boost library).</li>
  <li>DCGAN for color image generation of medium resolution (up to 162 pixels).</li>
  <li>Built-in internal web server for convenient model testing.</li>
  <li>Ready to use without extensive setup (pre-computed model provided).</li>
</ul>

<h2>Installation</h2>
<p>To install FFgan and recompile the program, follow these steps:</p>
<ol>
  <li>Ensure that you have the latest version of Dlib (version 19.24) installed.</li>
  <li>Make sure you have a compatible platform, such as Windows 10, and the Microsoft Visual Studio 2022 (64-bit, Version 17.6.0) compiler.</li>
  <li>Download the FFgan source code from the GitHub repository.</li>
  <li>Open the project in Microsoft Visual Studio.</li>
  <li>Recompile the project using the provided source files.</li>
</ol>
<p>Note: The provided synchronization file also allows you to recreate the model from scratch if needed.</p>

<p>The training process for FFgan involved utilizing a dataset of over 250,000 faces extracted from real photos captured from the internet. A dedicated crawler was created specifically for this purpose. The face extraction and alignment process used traditional techniques, including those demonstrated in the examples of the Dlib library.</p>

<p>FFgan designed from the example "dnn_dcgan_train_ex.cpp" (available at <a href="http://dlib.net/dnn_dcgan_train_ex.cpp.html">http://dlib.net/dnn_dcgan_train_ex.cpp.html</a>). The CNN model has been adapted for generating high-quality color images.</p>

<p>Generator model:</p>
<table>
  <thead>
    <tr>
      <th>Layer</th>
      <th>Output Shape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Input (1x100 Noise Tensor)</td>
      <td>(1, 100)</td>
    </tr>
    <tr>
      <td>ReLU/BatchNorm</td>
      <td>(4, 4, 512)</td>
    </tr>
    <tr>
      <td>ConvTranspose</td>
      <td>(10, 10, 256)</td>
    </tr>
    <tr>
      <td>ReLU/BatchNorm</td>
      <td>(20, 20, 128)</td>
    </tr>
    <tr>
      <td>ConvTranspose</td>
      <td>(40, 40, 128)</td>
    </tr>
    <tr>
      <td>ReLU/BatchNorm</td>
      <td>(80, 80, 64)</td>
    </tr>
    <tr>
      <td>ConvTranspose</td>
      <td>(162, 162, 3)</td>
    </tr>
    <tr>
      <td>Output</td>
      <td>(162, 162, 3)</td>
    </tr>    
    <tr>
      <td>Sigmoid</td>
      <td>(1, 1)</td>
    </tr>
    <tr>
      <td>FC</td>
      <td>(1, 1)</td>
    </tr>    
  </tbody>    
</table>

<p>Discriminator model:</p>
<table>
  <thead>
    <tr>
      <th>Layer</th>
      <th>Output Shape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Input: RGB Image</td>
      <td>(162, 162, 3)</td>
    </tr>
    <tr>
      <td>Convolution</td>
      <td>(160, 160, 3)</td>
    </tr>
    <tr>
      <td>LeakyReLU/BatchNorm/Dropout</td>
      <td>(160, 160, 3)</td>
    </tr>
    <tr>
      <td>Convolution</td>
      <td>(80, 80, 512)</td>
    </tr>
    <tr>
      <td>LeakyReLU/BatchNorm/Dropout</td>
      <td>(80, 80, 512)</td>
    </tr>
    <tr>
      <td>Convolution</td>
      <td>(78, 78, 256)</td>
    </tr>
    <tr>
      <td>LeakyReLU/BatchNorm/Dropout</td>
      <td>(78, 78, 256)</td>
    </tr>
    <tr>
      <td>Convolution</td>
      <td>(39, 39, 128)</td>
    </tr>
    <tr>
      <td>LeakyReLU/BatchNorm/Dropout</td>
      <td>(39, 39, 128)</td>
    </tr>
    <tr>
      <td>Convolution</td>
      <td>(19, 19, 128)</td>
    </tr>
    <tr>
      <td>LeakyReLU/BatchNorm/Dropout</td>
      <td>(19, 19, 128)</td>
    </tr>
    <tr>
      <td>Convolution</td>
      <td>(9, 9, 64)</td>
    </tr>
    <tr>
      <td>LeakyReLU/BatchNorm/Dropout</td>
      <td>(9, 9, 64)</td>
    </tr>
    <tr>
      <td>FullyConnected</td>
      <td>(1, 1)</td>
    </tr>
    <tr>
      <td>Output</td>
      <td>Real/Fake Classification</td>
    </tr>
  </tbody>
</table>

<h2>Usage</h2>
<p>
  The first example demonstrates training the FFgan model using the images in the specified directory. The second example shows how to generate a specified number of images automatically using the trained model. In this case, the command <code>FFgan --gen 10</code> generates 10 images.
</p>

<p>
  <code>FFgan --train &lt;directory&gt;</code>
</p>
<p>
  <strong>Description:</strong>
  <br>
  Trains or fine-tunes the FFgan model using the images provided in the specified directory. The directory should directly contain all the images or subdirectories containing the images. The images will be resized to 162x162 pixels during training. It is recommended that the images have a minimum size of 162 pixels on each side (note that the default face extraction modules in Dlib extract faces of 200 pixels on each side).
</p>
<p>
  <strong>Arguments:</strong>
</p>
<ul>
  <li><code>--train &lt;directory&gt;</code>: Specifies the directory containing the training images. The directory should directly contain the images or subdirectories containing the images.</li>
</ul>

<br><p>
  <code>FFgan --gen &lt;number&gt;</code>
</p>
<p>
  <strong>Description:</strong>
  <br>
  Generates a specified number of images automatically and displays them in a window. The program performs a test using the discriminator to check if the generated image is an acceptable "candidate." If not, it iterates a certain number of times to try to find a higher-quality image.
</p>
<p>
  <strong>Arguments:</strong>
</p>
<ul>
  <li><code>--gen &lt;number&gt;</code>: Generates a specified number of images automatically and displays them in a window. The program does not perform a test based on the specified number; instead, it allows the user to manually stop the generation process by closing the window where the generated faces are displayed or by using Ctrl+C in the execution console.</li>
</ul>

<br><p>
  <code>FFgan --web</code>
</p>
<p>
  <strong>Description:</strong>
  <br>
  Instantiates a local web server listening for requests on port 9190 and generates a face receiving a request from a Web browser.
</p>
<p>
  <strong>Arguments:</strong>
</p>
<ul>
  <li><code>--gen &lt;number&gt;</code>: Allows direct access to the generated images via a web interface. Users can access the generated images by navigating to <code>http://localhost:9190</code> in their web browser.</li>
</ul>

<h2>License</h2>
<p>
  This program is licensed under the GNU General Public License (GPL). The GPL grants users the freedom to use, modify, and distribute the software. However, commercial usage is not allowed under this license. It will also be strongly appreciated that any external usage of this program, especially in academic area, includes proper attribution to the author and his work. Please provide a reference <b>Cydral Technology</b> and acknowledge its contributions when using the FFgan program for research or other purposes.
</p>

<h2>Acknowledgments</h2>
<p>
  Special thanks to Davis E. King and all the contributors for the amazing Dlib library. Their dedication and hard work have made it possible to develop high-quality and efficient AI models using Dlib. We are grateful for the quality and speed of the AI models provided by Dlib, which greatly contribute to the success of the FFgan program.
</p>
