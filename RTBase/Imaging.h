#pragma once

#include "Core.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define __STDC_LIB_EXT1__
#include "stb_image_write.h"

// Stop warnings about buffer overruns if size is zero. Size should never be zero and if it is the code handles it.
#pragma warning( disable : 6386)

constexpr float texelScale = 1.0f / 255.0f;

class Texture
{
public:
	Colour* texels;
	float* alpha;
	int width;
	int height;
	int channels;
	void loadDefault()
	{
		width = 1;
		height = 1;
		channels = 3;
		texels = new Colour[1];
		texels[0] = Colour(1.0f, 1.0f, 1.0f);
	}
	void load(std::string filename)
	{
		alpha = NULL;
		if (filename.find(".hdr") != std::string::npos)
		{
			float* textureData = stbi_loadf(filename.c_str(), &width, &height, &channels, 0);
			if (width == 0 || height == 0)
			{
				loadDefault();
				return;
			}
			texels = new Colour[width * height];
			for (int i = 0; i < (width * height); i++)
			{
				texels[i] = Colour(textureData[i * channels], textureData[(i * channels) + 1], textureData[(i * channels) + 2]);
			}
			stbi_image_free(textureData);
			return;
		}
		unsigned char* textureData = stbi_load(filename.c_str(), &width, &height, &channels, 0);
		if (width == 0 || height == 0)
		{
			loadDefault();
			return;
		}
		texels = new Colour[width * height];
		for (int i = 0; i < (width * height); i++)
		{
			texels[i] = Colour(textureData[i * channels] / 255.0f, textureData[(i * channels) + 1] / 255.0f, textureData[(i * channels) + 2] / 255.0f);
		}
		if (channels == 4)
		{
			alpha = new float[width * height];
			for (int i = 0; i < (width * height); i++)
			{
				alpha[i] = textureData[(i * channels) + 3] / 255.0f;
			}
		}
		stbi_image_free(textureData);
	}
	Colour sample(const float tu, const float tv) const
	{
		Colour tex;
		float u = std::max(0.0f, fabsf(tu)) * width;
		float v = std::max(0.0f, fabsf(tv)) * height;
		int x = (int)floorf(u);
		int y = (int)floorf(v);
		float frac_u = u - x;
		float frac_v = v - y;
		float w0 = (1.0f - frac_u) * (1.0f - frac_v);
		float w1 = frac_u * (1.0f - frac_v);
		float w2 = (1.0f - frac_u) * frac_v;
		float w3 = frac_u * frac_v;
		x = x % width;
		y = y % height;
		Colour s[4];
		s[0] = texels[y * width + x];
		s[1] = texels[y * width + ((x + 1) % width)];
		s[2] = texels[((y + 1) % height) * width + x];
		s[3] = texels[((y + 1) % height) * width + ((x + 1) % width)];
		tex = (s[0] * w0) + (s[1] * w1) + (s[2] * w2) + (s[3] * w3);
		return tex;
	}
	float sampleAlpha(const float tu, const float tv) const
	{
		if (alpha == NULL)
		{
			return 1.0f;
		}
		float tex;
		float u = std::max(0.0f, fabsf(tu)) * width;
		float v = std::max(0.0f, fabsf(tv)) * height;
		int x = (int)floorf(u);
		int y = (int)floorf(v);
		float frac_u = u - x;
		float frac_v = v - y;
		float w0 = (1.0f - frac_u) * (1.0f - frac_v);
		float w1 = frac_u * (1.0f - frac_v);
		float w2 = (1.0f - frac_u) * frac_v;
		float w3 = frac_u * frac_v;
		x = x % width;
		y = y % height;
		float s[4];
		s[0] = alpha[y * width + x];
		s[1] = alpha[y * width + ((x + 1) % width)];
		s[2] = alpha[((y + 1) % height) * width + x];
		s[3] = alpha[((y + 1) % height) * width + ((x + 1) % width)];
		tex = (s[0] * w0) + (s[1] * w1) + (s[2] * w2) + (s[3] * w3);
		return tex;
	}
	~Texture()
	{
		delete[] texels;
		if (alpha != NULL)
		{
			delete alpha;
		}
	}
};

class ImageFilter
{
public:
	virtual float filter(const float x, const float y) const = 0;
	virtual int size() const = 0;
};

class BoxFilter : public ImageFilter
{
public:
	float filter(float x, float y) const
	{
		if (fabsf(x) <= 0.5f && fabs(y) <= 0.5f)
		{
			return 1.0f;
		}
		return 0;
	}
	int size() const
	{
		return 0;
	}
};

class GaussianFilter : public ImageFilter
{
public:
	float alpha;
	float radius;

	GaussianFilter(float _alpha, float _radius) {
		alpha = _alpha;
		radius = _radius;
	}

	float filter(float x, float y) const {
		float d = std::sqrt(x * x + y * y);
		return std::exp(-alpha * d) - std::exp(-alpha * radius * radius);
	}

	int size() const override{
		return std::ceil(radius);
	}
};

class MitchellFilter : public ImageFilter
{
public:
	float B;
	float C;

	MitchellFilter(float _B, float _C) {
		B = _B;
		C = _C;
	}

	float filter(float x) const {
		x = std::abs(x);
		if (x < 1.0f) {
			return ((12 - 9 * B - 6 * C) * std::pow(x, 3)
				+ (-18 + 12 * B + 6 * C) * std::pow(x, 2)
				+ (6 - 2 * B)) / 6.0f;
		}
		else if (x < 2.0f) {
			return ((-B * -6 * C) * std::pow(x, 3)
				+ (6 * B + 30 * C) * std::pow(x, 2)
				+ (-12 * B - 48 * C) * x
				+ (8 * B + 24 * C)) / 6.0f;
		}
		else {
			return 0;
		}
	}
};

class Film
{
public:
	Colour* film;
	Colour* albedoBuffer;
	Colour* normalBuffer;
	unsigned int width;
	unsigned int height;
	int SPP;
	ImageFilter* filter;
	void splat(const float x, const float y, const Colour& L)
	{
		// Code to splat a smaple with colour L into the image plane using an ImageFilter
		float filterWeights[25]; // Storage to cache weights 
		unsigned int indices[25]; // Store indices to minimize computations 
		unsigned int used = 0;
		float total = 0;
		int size = filter->size();
		for (int i = -size; i <= size; i++) {
			for (int j = -size; j <= size; j++) {
				int px = (int)x + j;
				int py = (int)y + i;
				if (px >= 0 && px < width && py >= 0 && py < height) {
					indices[used] = (py * width) + px;
					filterWeights[used] = filter->filter(j, i);
					total += filterWeights[used];
					used++;
				}
			}
		}
		for (int i = 0; i < used; i++) {
			film[indices[i]] = film[indices[i]] + (L * filterWeights[i] / total);
		}
	}

	void tonemap(int x, int y, unsigned char& r, unsigned char& g, unsigned char& b, float exposure = 1.0f)
	{
		int index = y * width + x;
		Colour pixel = film[index] * exposure / (float)SPP;
		r = std::min(powf(std::max(pixel.r, 0.0f), 1.0f / 2.2f) * 255, 255.0f);
		g = std::min(powf(std::max(pixel.g, 0.0f), 1.0f / 2.2f) * 255, 255.0f);
		b = std::min(powf(std::max(pixel.b, 0.0f), 1.0f / 2.2f) * 255, 255.0f);
		// Return a tonemapped pixel at coordinates x, y
	}
	// Do not change any code below this line
	void init(int _width, int _height, ImageFilter* _filter)
	{
		width = _width;
		height = _height;
		film = new Colour[width * height];
		albedoBuffer = new Colour[width * height];
		normalBuffer = new Colour[width * height];
		clear();
		filter = _filter;
	}
	void clear()
	{
		memset(film, 0, width * height * sizeof(Colour));
		memset(albedoBuffer, 0, width * height * sizeof(Colour));
		memset(normalBuffer, 0, width * height * sizeof(Colour));
		SPP = 0;
	}
	void incrementSPP()
	{
		SPP++;
	}
	void save(std::string filename)
	{
		Colour* hdrpixels = new Colour[width * height];
		for (unsigned int i = 0; i < (width * height); i++)
		{
			hdrpixels[i] = film[i] / (float)SPP;
		}
		stbi_write_hdr(filename.c_str(), width, height, 3, (float*)hdrpixels);
		delete[] hdrpixels;
	}
};