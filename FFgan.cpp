#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <boost/beast.hpp>
#include <boost/asio.hpp>

#include <boost/filesystem.hpp>
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/matrix.h>
#include <dlib/image_processing.h>
#include <jpeglib.h>

using namespace std;
using namespace dlib;
namespace fs = boost::filesystem;
using gray_pixel = uint8_t;
using namespace boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = boost::asio::ip::tcp;

const size_t default_image_size = 162;
const size_t default_display_image_size = 90;

// Some helper definitions for the noise generation
const size_t noise_size = 100;
using noise_t = std::array<matrix<float, 1, 1>, noise_size>;
noise_t make_noise(dlib::rand& rnd) {
    noise_t noise;
    for (auto& n : noise) n = rnd.get_random_gaussian();
    return noise;
}

// A convolution with custom padding
template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
using conp = add_layer<con_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

// A transposed convolution to with custom padding
template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
using contp = add_layer<cont_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

// The generator is made of a bunch of deconvolutional layers.  Its input is a 1 x 1 x k noise
// tensor, and the output is the generated image.  The loss layer does not matter for the
// training, we just stack a compatible one on top to be able to have a () operator on the
// generator.
//
// For 162x162 images
using generator_type = loss_binary_log_per_pixel<fc_no_bias<1,
    sig<contp<3, 4, 2, 0,               // 162
    relu<bn_con<contp<64, 4, 2, 1,      // 80
    relu<bn_con<contp<128, 4, 2, 1,     // 40
    relu<bn_con<contp<128, 4, 2, 1,     // 20
    relu<bn_con<contp<256, 3, 3, 1,     // 10
    relu<bn_con<contp<512, 4, 1, 0,     // 4
    input<noise_t>                      // 1
    >>>>>>>>>>>>>>>>>>>;
using display_generator_type = loss_binary_log_per_pixel<fc_no_bias<1,
    sig<contp<3, 4, 2, 0,               // 162
    relu<affine<contp<64, 4, 2, 1,      // 80
    relu<affine<contp<128, 4, 2, 1,     // 40
    relu<affine<contp<128, 4, 2, 1,     // 20
    relu<affine<contp<256, 3, 3, 1,     // 10
    relu<affine<contp<512, 4, 1, 0,     // 4
    input<noise_t>                      // 1
    >>>>>>>>>>>>>>>>>>>;

// Now, let's proceed to define the discriminator, whose role will be to decide whether an
// image is fake or not.
// For 162x162 images
using discriminator_type = loss_binary_log<fc<1,
    conp<3, 3, 1, 0,                                // 1
    dropout<leaky_relu<bn_con<conp<512, 4, 2, 1,    // 3
    dropout<leaky_relu<bn_con<conp<256, 3, 3, 1,    // 7
    dropout<leaky_relu<bn_con<conp<128, 4, 2, 1,    // 20
    dropout<leaky_relu<bn_con<conp<128, 4, 2, 1,    // 40
    dropout<leaky_relu<conp<64, 4, 2, 0,            // 80
    input<matrix<rgb_pixel>>                        // 162
    >>>>>>>>>>>>>>>>>>>>>>;
using display_discriminator_type = loss_binary_log<fc<1,
    conp<3, 3, 1, 0,
    dropout<leaky_relu<affine<conp<512, 4, 2, 1,
    dropout<leaky_relu<affine<conp<256, 3, 3, 1,
    dropout<leaky_relu<affine<conp<128, 4, 2, 1,
    dropout<leaky_relu<affine<conp<128, 4, 2, 1,
    dropout<leaky_relu<conp<64, 4, 2, 0,
    input<matrix<rgb_pixel>>
    >>>>>>>>>>>>>>>>>>>>>>;

std::atomic<bool> g_interrupted = false;
std::atomic<bool> g_web_server = false;
BOOL WINAPI CtrlHandler(DWORD ctrlType) {
    if (ctrlType == CTRL_C_EVENT) {
        g_interrupted = true;
        if (g_web_server) {
            try {
                boost::asio::io_service io_service;
                tcp::resolver resolver(io_service);
                tcp::resolver::iterator endpoints = resolver.resolve("localhost", "9190");
                tcp::socket stop_socket(io_service);
                boost::asio::connect(stop_socket, endpoints);
                http::request<http::empty_body> stop_request;
                stop_request.method(http::verb::get);
                stop_request.target("/stop");
                http::write(stop_socket, stop_request);
                boost::asio::streambuf response;
                boost::asio::read_until(stop_socket, response, "\r\n");
                std::istream response_stream(&response);
                std::string http_version;
                response_stream >> http_version;
                unsigned int status_code;
                response_stream >> status_code;
                std::string status_message;
                std::getline(response_stream, status_message);
                if (response_stream && http_version.substr(0, 5) != "HTTP/" && status_code == 200)
                {
                    boost::asio::read_until(stop_socket, response, "\r\n\r\n");
                    std::string header;
                    while (std::getline(response_stream, header) && header != "\r");
                    boost::system::error_code error;
                    while (boost::asio::read(stop_socket, response, boost::asio::transfer_at_least(1), error));
                }
            } catch (std::exception& e) {
                std::cout << "Exception: " << e.what() << "\n";
            }
        }
        return TRUE;
    }
    return FALSE;
}

// RGB to grayscale image conversion.
void rgb_image_to_grayscale_image(const matrix<dlib::rgb_pixel>& rgb_image, matrix<gray_pixel>& gray_image) {
    gray_image.set_size(rgb_image.nr(), rgb_image.nc());
    std::transform(rgb_image.begin(), rgb_image.end(), gray_image.begin(),
        [](rgb_pixel a) {return gray_pixel(a.red * 0.299f + a.green * 0.587f + a.blue * 0.114f); });
}
void grayscale_image_to_rgb_image(const dlib::matrix<gray_pixel>& r_channel, const dlib::matrix<gray_pixel>& g_channel, const dlib::matrix<gray_pixel>& b_channel, dlib::matrix<dlib::rgb_pixel>& rgb_image) {
    rgb_image.set_size(r_channel.nr(), r_channel.nc());
    for (long r = 0; r < rgb_image.nr(); ++r) {
        for (long c = 0; c < rgb_image.nc(); ++c) {
            rgb_image(r, c).red   = r_channel(r, c);
            rgb_image(r, c).green = g_channel(r, c);
            rgb_image(r, c).blue  = b_channel(r, c);
        }
    }
}

// Helper function to resize grayscale or color image.
template <typename pixel_type>
void resize_inplace(matrix<pixel_type>& inout, long size = default_image_size) {
    if (inout.nr() != size || inout.nc() != size) {
        matrix<pixel_type> mem_img;
        mem_img.set_size(size, size);
        resize_image(inout, mem_img);
        inout = mem_img;
    }
}
template <typename pixel_type>
void resize_images(std::vector<matrix<pixel_type>>& images, long new_size = default_image_size) {
    for (auto& image : images) resize_inplace(image, new_size);
}

// Some helper functions to generate and get the images from the generator
std::mutex gen_mutex;  // Déclaration du mutex
template <typename pixel_type>
matrix<pixel_type> generate_image(generator_type& net, const noise_t& noise) {
    net(noise);
    matrix<pixel_type> image;
    if constexpr (std::is_same_v<pixel_type, gray_pixel>) {
        matrix<float> output = image_plane(layer<2>(net).get_output(), 0, 0);
        for (long r = 0; r < output.nr(); ++r) {
            for (long c = 0; c < output.nc(); ++c) {
                output(r, c) = __max(0.0f, __min(1.0f, output(r, c)));
            }
        }
        assign_image(image, 255 * output);
    } else {
        matrix<float> output_r = image_plane(layer<2>(net).get_output(), 0, 0);
        matrix<float> output_g = image_plane(layer<2>(net).get_output(), 0, 1);
        matrix<float> output_b = image_plane(layer<2>(net).get_output(), 0, 2);
        for (long r = 0; r < output_r.nr(); ++r) {
            for (long c = 0; c < output_r.nc(); ++c) {
                output_r(r, c) = __max(0.0f, __min(1.0f, output_r(r, c)));
                output_g(r, c) = __max(0.0f, __min(1.0f, output_g(r, c)));
                output_b(r, c) = __max(0.0f, __min(1.0f, output_b(r, c)));
            }
        }
        matrix<gray_pixel> r_channel, g_channel, b_channel;
        assign_image(r_channel, 255 * output_r);
        assign_image(g_channel, 255 * output_g);
        assign_image(b_channel, 255 * output_b);
        grayscale_image_to_rgb_image(r_channel, g_channel, b_channel, image);
    }
    return image;
}
template <typename pixel_type>
matrix<pixel_type> generate_image_for_display(display_generator_type& net, const noise_t& noise) {
    net(noise);
    matrix<pixel_type> image;
    if constexpr (std::is_same_v<pixel_type, gray_pixel>) {
        const matrix<float> output = image_plane(layer<2>(net).get_output(), 0, 0);
        assign_image(image, 255 * output);
    } else {
        matrix<float> output_r = image_plane(layer<2>(net).get_output(), 0, 0);
        matrix<float> output_g = image_plane(layer<2>(net).get_output(), 0, 1);
        matrix<float> output_b = image_plane(layer<2>(net).get_output(), 0, 2);
        matrix<gray_pixel> r_channel, g_channel, b_channel;
        assign_image(r_channel, 255 * output_r);
        assign_image(g_channel, 255 * output_g);
        assign_image(b_channel, 255 * output_b);
        grayscale_image_to_rgb_image(r_channel, g_channel, b_channel, image);
    }
    return image;
}

template <typename pixel_type>
std::vector<matrix<pixel_type>> get_generated_images(const tensor& out) {
    std::vector<matrix<pixel_type>> images;
    matrix<pixel_type> image;
    if constexpr (std::is_same_v<pixel_type, gray_pixel>) {
        for (long n = 0; n < out.num_samples(); ++n)
        {
            matrix<float> output = image_plane(out, n, 0);
            assign_image(image, 255 * output);
            images.push_back(image);
        }
    } else {
        for (long n = 0; n < out.num_samples(); ++n) {
            matrix<float> r_output = image_plane(out, n, 0);
            matrix<float> g_output = image_plane(out, n, 1);
            matrix<float> b_output = image_plane(out, n, 2);
            matrix<gray_pixel> r_channel, g_channel, b_channel;
            assign_image(r_channel, 255 * r_output);
            assign_image(g_channel, 255 * g_output);
            assign_image(b_channel, 255 * b_output);
            grayscale_image_to_rgb_image(r_channel, g_channel, b_channel, image);
            images.push_back(image);
        }
    }
    return images;
}

// Function to load and resize images from a directory recursively
template <typename pixel_type>
void load_images_from_directory(const fs::path& directory, std::vector<matrix<pixel_type>>& images, const int size, const int limit = 1e+6) {
    fs::recursive_directory_iterator end_itr;
    images.clear();
    for (fs::recursive_directory_iterator itr(directory); itr != end_itr && images.size() < limit && !g_interrupted; ++itr) {
        if (!fs::is_regular_file(itr->status()) || itr->path().extension() != ".jpg") continue;
        matrix<rgb_pixel> image;
        try { load_image(image, itr->path().string()); }
        catch (...) {
            cerr << "Error during image loading: " << itr->path().string() << endl;
        }
        resize_inplace(image, size);

        if constexpr (std::is_same_v<pixel_type, gray_pixel>) {
            matrix<gray_pixel> gray_image;
            rgb_image_to_grayscale_image(image, gray_image);
            images.push_back(std::move(gray_image));
        }
        else {
            images.push_back(std::move(image));
        }
        if (images.size() % 10000 == 0) cout << ".";
    }
}
void load_images_from_directory(const fs::path& directory, std::vector<fs::path>& images, const int limit = 1e+6) {
    fs::recursive_directory_iterator end_itr;
    images.clear();
    for (fs::recursive_directory_iterator itr(directory); itr != end_itr && images.size() < limit && !g_interrupted; ++itr)
    {
        if (!fs::is_regular_file(itr->status()) || itr->path().extension() != ".jpg") continue;
        images.push_back(itr->path());
        if (images.size() % 5000 == 0) cout << ".";
    }
}

std::vector<fs::path> get_shuffled_paths(const std::vector<fs::path>& path_images) {
    std::vector<fs::path> shuffled_paths = path_images;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(shuffled_paths.begin(), shuffled_paths.end(), g);
    return shuffled_paths;
}

// Manage web requests
static const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
string base64_encode(unsigned char const* buf, unsigned int bufLen) {
    string ret;
    int i = 0, j = 0;
    unsigned char char_array_3[3], char_array_4[4];
    while (bufLen--) {
        char_array_3[i++] = *(buf++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            for (i = 0; (i < 4); i++) ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }
    if (i) {
        for (j = i; j < 3; j++) char_array_3[j] = '\0';
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;
        for (j = 0; (j < i + 1); j++) ret += base64_chars[char_array_4[j]];
        while ((i++ < 3)) ret += '=';
    }
    return ret;
}
bool set_load_jpeg_buffer(dlib::matrix<dlib::rgb_pixel>& in_img, std::vector<unsigned char>& compressed) {
    compressed.resize(in_img.nc() * in_img.nr() * 3);
    unsigned char* mem = compressed.data();
    unsigned long mem_size = compressed.size();
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_mem_dest(&cinfo, &mem, &mem_size);

    // Setting the parameters of the output file here
    cinfo.image_width = in_img.nc();
    cinfo.image_height = in_img.nr();
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    // Default compression parameters, we shouldn't be worried about these
    jpeg_set_defaults(&cinfo);
    // Now do the compression
    jpeg_set_quality(&cinfo, 95, true);
    jpeg_start_compress(&cinfo, TRUE);

    // Like reading a file, this time write one row at a time
    JSAMPROW row_pointer[1];
    const unsigned int row_stride = cinfo.image_width * cinfo.input_components;
    std::vector<unsigned char> buf(cinfo.image_height * row_stride);
    unsigned char* buf2 = &buf[0];
    dlib::rgb_pixel pixel;
    for (size_t r = 0; r < cinfo.image_height; r++) {
        for (size_t c = 0; c < cinfo.image_width; c++) {
            pixel = in_img(r, c);
            *(buf2++) = pixel.red;
            *(buf2++) = pixel.green;
            *(buf2++) = pixel.blue;
        }
    }
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = &buf[0] + cinfo.next_scanline * row_stride;
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }
    // Similar to read file, clean up after we're done compressing
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    // Copy compressed data and release the memory
    if (mem_size == 0) compressed.clear();
    else compressed.resize(mem_size);

    return compressed.size();
}
void handle_request(display_generator_type& gen, display_discriminator_type& disc, dlib::rand& rnd, const http::request<http::string_body>& req, http::response<http::string_body>& res) {
    try {
        if (req.method() == http::verb::get && (req.target().empty() || req.target() == "/get_raw_image" || req.target() == "/get_image" || req.target() == "/")) {
            const bool send_raw_data = (req.target() == "/get_raw_image");
            matrix<rgb_pixel> gen_image;
            bool is_real = false;
            size_t current_image = 0, target_image_size = 150;            
            while (!is_real && current_image++ < 10 && !g_interrupted) {
                std::lock_guard<std::mutex> lock(gen_mutex);
                gen_image = generate_image_for_display<rgb_pixel>(gen, make_noise(rnd));
                is_real = (send_raw_data || (disc(gen_image) > 0));
            }            
            resize_inplace(gen_image, target_image_size);
            std::vector<unsigned char> compressed;
            set_load_jpeg_buffer(gen_image, compressed);

            // HTML dynamically created
            std::string html;
            if (send_raw_data) {
                html += "data:image/jpeg;base64," + base64_encode(reinterpret_cast<const unsigned char*>(compressed.data()), compressed.size());
            } else {
                html += "<html><head>";
                html += "<style>";
                html += "body { text-align: center; }";
                html += "h1 { color: #333; }";
                html += "img { cursor: pointer; }";
                html += "</style>";
                html += "</head><body>";
                html += "<h1>FAKES - Generated image</h1>";
                html += "<p>GANs are used to create synthetic data, such as realistic faces of non-existent individuals, by training a generator to produce data that can deceive a discriminator:</p>";
                html += "<img src=\"data:image/jpeg;base64," + base64_encode(reinterpret_cast<const unsigned char*>(compressed.data()), compressed.size()) + "\" alt=\"Generated Image\" onclick=\"location.reload();\">";
                html += "<br><br>";
                html += "<button onclick=\"location.reload();\">Regenerate</button>";
                html += "</body></html>";
            }

            // Send back the response
            res.result(http::status::ok);
            res.set(http::field::content_type, send_raw_data ? "text/plain" : "text/html");
            res.body() = std::move(html);
        } else {
            res.result(http::status::not_found);
            res.set(http::field::content_type, "text/plain");
            res.body() = std::move(std::string("Page not found"));
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        res.result(http::status::internal_server_error);
        res.set(http::field::content_type, "text/plain");
        res.body() = std::move(std::string("Internal Server Error"));
    }
}

int main(int argc, char** argv) try {
    if (argc < 2) {
        std::cout << "Usage: FFgan --train <directory>, --gen <number> or --web" << std::endl;
        return EXIT_FAILURE;
    }
    std::string option = argv[1];
    std::srand(std::time(nullptr));
    dlib::rand rnd(std::rand());
    set_dnn_prefer_smallest_algorithms();
    size_t epoch = 0, iteration = 0;

    if (option == "--train") {
        if (argc < 3) {
            std::cout << "Please provide the directory for training images" << std::endl;
            return 1;
        }
        SetConsoleCtrlHandler(CtrlHandler, TRUE);
        std::string directory = argv[2];
        std::vector<fs::path> training_images, pool_images;
        load_images_from_directory(directory, training_images);
        cout << endl << training_images.size() << " images found in <" << directory << ">" << endl;
        if (training_images.size() == 0) return EXIT_FAILURE;

        // Instantiate both generator and discriminator
        generator_type generator;
        discriminator_type discriminator;
        // Setup all leaky_relu_ layers in the discriminator to have alpha = 0.2
        visit_computational_layers(discriminator, [](leaky_relu_& l) { l = leaky_relu_(0.2); });
        // Remove the bias learning from all bn_ inputs in both networks
        disable_duplicative_biases(generator);
        disable_duplicative_biases(discriminator);
        // Forward random noise so that we see the tensor size at each layer
        discriminator(generate_image<rgb_pixel>(generator, make_noise(rnd)));
        cout << "generator (" << count_parameters(generator) << " parameters)" << endl;
        cout << generator << endl;
        cout << "discriminator (" << count_parameters(discriminator) << " parameters)" << endl;
        cout << discriminator << endl;

        // The solvers for the generator and discriminator networks
        std::vector<adam> g_solvers(generator.num_computational_layers, adam(0, 0.5, 0.999));
        std::vector<adam> d_solvers(discriminator.num_computational_layers, adam(0, 0.5, 0.999));
        double learning_rate = 2e-4;

        // Resume training from last sync file
        if (file_exists("dcgan_162x162_synth_faces.sync")) {
            deserialize("dcgan_162x162_synth_faces.sync") >> generator >> discriminator >> iteration >> epoch;
        } else if (file_exists("dcgan_162x162_synth_faces.dnn")) {
            deserialize("dcgan_162x162_synth_faces.dnn") >> generator >> discriminator;
        }

        const size_t minibatch_size = 64;        
        const std::vector<float> real_labels(minibatch_size, 1);
        const std::vector<float> fake_labels(minibatch_size, -1);        
        resizable_tensor real_samples_tensor, fake_samples_tensor, noises_tensor;
        running_stats<double> g_loss, d_loss;
        dlib::image_window win;
        while (!g_interrupted) {
            // Train the discriminator with real images            
            std::vector<matrix<rgb_pixel>> real_samples;            
            while (real_samples.size() < minibatch_size) {
                if (pool_images.size() == 0) {
                    pool_images = get_shuffled_paths(training_images);
                    epoch++;
                }
                fs::path p_img = pool_images.back();
                pool_images.pop_back();
                matrix<rgb_pixel> tmp_image;
                try { load_image(tmp_image, p_img.string()); }
                catch (...) {
                    cerr << "Error during image loading: " << p_img.string() << endl;
                    continue;
                }
                resize_inplace(tmp_image, default_image_size);
                real_samples.push_back(tmp_image);
            }            
            discriminator.to_tensor(real_samples.begin(), real_samples.end(), real_samples_tensor);
            discriminator.forward(real_samples_tensor);
            d_loss.add(discriminator.compute_loss(real_samples_tensor, real_labels.begin()));
            discriminator.back_propagate_error(real_samples_tensor);
            discriminator.update_parameters(d_solvers, learning_rate);

            // Train the discriminator with fake images
            // 1. Generate some random noise
            std::vector<noise_t> noises;
            while (noises.size() < minibatch_size) noises.push_back(make_noise(rnd));
            // 2. Convert noises into a tensor
            generator.to_tensor(noises.begin(), noises.end(), noises_tensor);
            // 3. Forward the noise through the network and convert the outputs into images
            generator.forward(noises_tensor);
            auto fake_samples = get_generated_images<rgb_pixel>(layer<2>(generator).get_output());
            // 4. Finally train the discriminator
            discriminator.to_tensor(fake_samples.begin(), fake_samples.end(), fake_samples_tensor);
            discriminator.forward(fake_samples_tensor);
            d_loss.add(discriminator.compute_loss(fake_samples_tensor, fake_labels.begin()));
            discriminator.back_propagate_error(fake_samples_tensor);
            discriminator.update_parameters(d_solvers, learning_rate);

            // Train the generator
            // This part is the essence of the Generative Adversarial Networks.  Until now, we have
            // just trained a binary classifier that the generator is not aware of.  But now, the
            // discriminator is going to give feedback to the generator on how it should update
            // itself to generate more realistic images.

            // Forward the fake samples and compute the loss with real labels
            g_loss.add(discriminator.compute_loss(fake_samples_tensor, real_labels.begin()));
            // Back propagate the error to fill the final data gradient
            discriminator.back_propagate_error(fake_samples_tensor);
            // Get the gradient that will tell the generator how to update itself
            const tensor& d_grad = discriminator.get_final_data_gradient();
            layer<2>(generator).back_propagate_error(noises_tensor, d_grad);
            generator.update_parameters(g_solvers, learning_rate);

            // At some point, we should see that the generated images start looking like samples
            if (++iteration % 20 == 0) { // Display                
                for (auto& image : fake_samples) resize_inplace(image, default_display_image_size);
                win.set_image(tile_images(fake_samples));
                win.set_title("FAKES - DCGAN step#: " + to_string(iteration));
            }
            if (iteration % 100 == 0) { // Progress                
                std::cout <<
                    "epoch#: " << epoch <<
                    "\tstep#: " << iteration <<
                    "\tdiscriminator loss: " << d_loss.mean() * 2 <<
                    "\tgenerator loss: " << g_loss.mean() << '\n';
            }
            if (iteration % 1000 == 0) { // Checkpoint                
                serialize("dcgan_162x162_synth_faces.sync") << generator << discriminator << iteration << epoch;
                d_loss.clear();
                g_loss.clear();
            }
        }

        // Once the training has finished, we don't need the discriminator any more. We just keep the generator.
        serialize("dcgan_162x162_synth_faces.sync") << generator << discriminator << iteration << epoch;
        generator.clean();
        discriminator.clean();
        serialize("dcgan_162x162_synth_faces.dnn") << generator << discriminator;
    } else if (option == "--gen") {
        if (argc < 3) {
            std::cout << "Please provide the total number of faked images to generate" << std::endl;
            return 1;
        }
        SetConsoleCtrlHandler(CtrlHandler, TRUE);
        char* end_ptr;
        size_t max_images = strtoul(argv[2], &end_ptr, 10);

        // Instantiate both generator and discriminator
        cout << "Loading model... ";
        display_generator_type generator;
        display_discriminator_type discriminator;
        if (file_exists("dcgan_162x162_synth_faces.dnn")) deserialize("dcgan_162x162_synth_faces.dnn") >> generator >> discriminator;
        cout << "done" << endl;

        dlib::image_window win;
        const string suffix = "gen_face_";
        win.set_title("FAKES - Generated image");
        matrix<rgb_pixel> gen_image;
        bool is_real;
        size_t current_image, total_images = 0;
        size_t target_image_size = 150;
        while (!win.is_closed() && total_images++ < max_images && !g_interrupted) {
            current_image = 0;
            is_real = false;
            while (!is_real && current_image++ < 30) {
                std::lock_guard<std::mutex> lock(gen_mutex);
                gen_image = generate_image_for_display<rgb_pixel>(generator, make_noise(rnd));
                is_real = (discriminator(gen_image) > 0);
            }         
            resize_inplace(gen_image, target_image_size);
            save_jpeg(gen_image, suffix + to_string(current_image) + string(".jpg"), 95);
            win.set_image(gen_image);
            sleep(500);
        }
    } else if (option == "--web") {        
        // Instantiate both generator and discriminator
        cout << "Loading model... ";
        display_generator_type generator;
        display_discriminator_type discriminator;
        if (file_exists("dcgan_162x162_synth_faces.dnn")) deserialize("dcgan_162x162_synth_faces.dnn") >> generator >> discriminator;
        cout << "done" << endl;

        // Instantiate the Web server
        g_web_server = true;
        SetConsoleCtrlHandler(CtrlHandler, TRUE);
        boost::asio::io_context io_context;            
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 9190));
        cout << "Listening on <http://localhost:9190>..." << endl;            

        while (!g_interrupted) {
            try {
                tcp::socket socket(io_context);
                acceptor.accept(socket);

                beast::flat_buffer buffer;
                http::request<http::string_body> request;
                http::read(socket, buffer, request);

                http::response<http::string_body> response;
                handle_request(generator, discriminator, rnd, request, response);
                http::write(socket, response);
                socket.close();
            }
            catch (const std::exception& e) {
                std::cerr << "Exception: " << e.what() << std::endl;
            }
        }
    } else {
        std::cout << "Invalid option. Usage: FFgan --train <directory>, --gen <number> or --web" << std::endl;
    }
    return EXIT_SUCCESS;
} catch (exception& e) {
    cout << e.what() << endl;
    return EXIT_FAILURE;
}
