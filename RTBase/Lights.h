#pragma once

#include "Core.h"
#include "Geometry.h"
#include "Materials.h"
#include "Sampling.h"

#pragma warning( disable : 4244)

class SceneBounds
{
public:
	Vec3 sceneCentre;
	float sceneRadius;
};

class Light
{
public:
	virtual Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf) = 0;
	virtual Colour evaluate(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual bool isArea() = 0;
	virtual Vec3 normal(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual float totalIntegratedPower() = 0;
	virtual Vec3 samplePositionFromLight(Sampler* sampler, float& pdf) = 0;
	virtual Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf) = 0;
};

class AreaLight : public Light
{
public:
	Triangle* triangle = NULL;
	Colour emission;
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf)
	{
		emittedColour = emission;
		return triangle->sample(sampler, pdf);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		if (Dot(wi, triangle->gNormal()) < 0)
		{
			return emission;
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return 1.0f / triangle->area;
	}
	bool isArea()
	{
		return true;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return triangle->gNormal();
	}
	float totalIntegratedPower()
	{
		return (triangle->area * emission.Lum());
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		return triangle->sample(sampler, pdf);
	}
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		// Add code to sample a direction from the light
		Vec3 wi = Vec3(0, 0, 1);
		pdf = 1.0f;
		Frame frame;
		frame.fromVector(triangle->gNormal());
		return frame.toWorld(wi);
	}
};

class BackgroundColour : public Light
{
public:
	Colour emission;
	BackgroundColour(Colour _emission)
	{
		emission = _emission;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		reflectedColour = emission;
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		return emission;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return SamplingDistributions::uniformSpherePDF(wi);
	}
	bool isArea()
	{
		return false;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return -wi;
	}
	float totalIntegratedPower()
	{
		return emission.Lum() * 4.0f * M_PI;
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 p = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		p = p * use<SceneBounds>().sceneRadius;
		p = p + use<SceneBounds>().sceneCentre;
		pdf = 4 * M_PI * use<SceneBounds>().sceneRadius * use<SceneBounds>().sceneRadius;
		return p;
	}
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		return wi;
	}
};

class EnvMapSampler {
public:
	int width, height;
	std::vector<float> marginalCDF;
	std::vector<std::vector<float>> conditionalCDF;
	Texture* env;

	EnvMapSampler(Texture* _env) : env(_env), width(_env->width), height(_env->height) {
		build();
	}

	void build() {
		conditionalCDF.resize(height);
		marginalCDF.resize(height);
		float total = 0.0f;

		for (int y = 0; y < height; ++y) {
			float sinTheta = sinf(M_PI * (y + 0.5f) / height);
			conditionalCDF[y].resize(width);
			float rowSum = 0.0f;

			for (int x = 0; x < width; ++x) {
				Colour texel = env->texels[y * width + x];
				float luminance = texel.Lum() * sinTheta;
				conditionalCDF[y][x] = luminance + (x > 0 ? conditionalCDF[y][x - 1] : 0.0f);
				rowSum += luminance;
			}

			// normalize conditional CDF
			for (int x = 0; x < width; ++x) {
				conditionalCDF[y][x] /= conditionalCDF[y][width - 1];
			}

			marginalCDF[y] = rowSum + (y > 0 ? marginalCDF[y - 1] : 0.0f);
			total += rowSum;
		}

		// normalize marginal CDF
		for (int y = 0; y < height; ++y) {
			marginalCDF[y] /= marginalCDF[height - 1];
		}
	}

	Vec3 sample(float u1, float u2, float& pdf)
	{
		// Sample row (v direction) using marginal CDF
		int y = 0;
		for (; y < height - 1; ++y)
		{
			if (u1 < marginalCDF[y]) break;
		}

		// Map back to row probability range
		float v0 = y == 0 ? 0.0f : marginalCDF[y - 1];
		float v1 = marginalCDF[y];
		float dv = (u1 - v0) / (v1 - v0 + 1e-6f);

		// Sample column (u direction) using conditional CDF
		int x = 0;
		for (; x < width - 1; ++x)
		{
			if (u2 < conditionalCDF[y][x]) break;
		}

		float u0 = x == 0 ? 0.0f : conditionalCDF[y][x - 1];
		float u1cdf = conditionalCDF[y][x];
		float du = (u2 - u0) / (u1cdf - u0 + 1e-6f);

		// convert to u,v in [0,1]
		float u = (x + du) / (float)width;
		float v = (y + dv) / (float)height;

		// latitude -> wi
		float theta = v * M_PI;
		float phi = u * 2.0f * M_PI;

		float sinTheta = sinf(theta);
		float cosTheta = cosf(theta);
		float sinPhi = sinf(phi);
		float cosPhi = cosf(phi);

		Vec3 wi = Vec3(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);

		// compute pdf
		float sinT = std::max(sinTheta, 1e-6f);
		float mapPDF = conditionalCDF[y][x] * marginalCDF[y];
		pdf = (width * height * mapPDF) / (2.0f * M_PI * M_PI * sinT);

		return wi;
	}

	Colour evaluate(const Vec3& dir)
	{
		float theta = acosf(clamp(dir.y, -1.0f, 1.0f)); // θ in [0, π]
		float phi = atan2f(dir.z, dir.x);              // φ in [-π, π]
		if (phi < 0.0f) phi += 2.0f * M_PI;

		float u = phi / (2.0f * M_PI);// [0, 1]
		float v = theta / M_PI;// [0, 1]

		return env->sample(u, v);
	}

	float PDF(const Vec3& dir)
	{
		float theta = acosf(clamp(dir.y, -1.0f, 1.0f));
		float phi = atan2f(dir.z, dir.x);
		if (phi < 0.0f) phi += 2.0f * M_PI;
		float u = phi / (2.0f * M_PI);
		float v = theta / M_PI;
		int x = clamp(int(u * width), 0, width - 1);
		int y = clamp(int(v * height), 0, height - 1);
		float sinTheta = std::max(sinf(theta), 1e-6f);
		float mapPDF = conditionalCDF[y][x] * marginalCDF[y];
		return (width * height * mapPDF) / (2.0f * M_PI * M_PI * sinTheta);
	}
};

class EnvironmentMap : public Light
{
public:
	Texture* env;
	EnvMapSampler sampler;

	EnvironmentMap(Texture* _env) : env(_env), sampler(_env) {}

	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 wi = this->sampler.sample(sampler->next(), sampler->next(), pdf);
		reflectedColour = evaluate(shadingData, wi); //evaluate radiance
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		//float u = atan2f(wi.z, wi.x);
		//u = (u < 0.0f) ? u + (2.0f * M_PI) : u;
		//u = u / (2.0f * M_PI);
		//float v = acosf(wi.y) / M_PI;
		//return env->sample(u, v);
		return sampler.evaluate(wi);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return sampler.PDF(wi);
	}
	bool isArea()
	{
		return false;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return -wi;
	}
	float totalIntegratedPower()
	{
		float total = 0;
		for (int i = 0; i < env->height; i++)
		{
			float st = sinf(((float)i / (float)env->height) * M_PI);
			for (int n = 0; n < env->width; n++)
			{
				total += (env->texels[(i * env->width) + n].Lum() * st);
			}
		}
		total = total / (float)(env->width * env->height);
		return total * 4.0f * M_PI;
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		// Samples a point on the bounding sphere of the scene. Feel free to improve this.
		Vec3 p = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		p = p * use<SceneBounds>().sceneRadius;
		p = p + use<SceneBounds>().sceneCentre;
		pdf = 1.0f / (4 * M_PI * SQ(use<SceneBounds>().sceneRadius));
		return p;
	}
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		// Replace this tabulated sampling of environment maps
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		return wi;
	}
};