#pragma once

#include "Core.h"
#include "Sampling.h"
#include "Geometry.h"
#include "Imaging.h"
#include "Materials.h"
#include "Lights.h"
#include "Scene.h"
#include "GamesEngineeringBase.h"
#include <thread>
#include <functional>

class RayTracer
{
public:
	Scene* scene;
	GamesEngineeringBase::Window* canvas;
	Film* film;
	MTRandom* samplers;
	std::thread** threads;
	int numProcs;

	struct Tile {
		int x0, y0, x1, y1;
		int spp; //current sampling count
		bool converged; 
		Colour mean;
		Colour M2; // cumulative square deviation
		Tile(int _x0, int _y0, int _x1, int _y1)
			: x0(_x0), y0(_y0), x1(_x1), y1(_y1), spp(0), converged(false), mean(0, 0, 0), M2(0, 0, 0) {}
	};

	void init(Scene* _scene, GamesEngineeringBase::Window* _canvas)
	{
		scene = _scene;
		canvas = _canvas;
		film = new Film();
		film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new GaussianFilter(1.0f, 1.0f));
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		numProcs = sysInfo.dwNumberOfProcessors;
		threads = new std::thread * [numProcs];
		samplers = new MTRandom[numProcs];
		clear();
	}
	void clear()
	{
		film->clear();
	}

	Colour computeDirect(ShadingData shadingData, Sampler* sampler)
	{
		if (shadingData.bsdf->isPureSpecular() == true)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		// Sample a light
		float pmf;
		Light* light = scene->sampleLight(sampler, pmf);
		// Sample a point on the light
		float pdf;
		Colour emitted;
		Vec3 p = light->sample(shadingData, sampler, emitted, pdf);
		if (light->isArea())
		{
			// Calculate GTerm
			Vec3 wi = p - shadingData.x;
			float l = wi.lengthSq();
			wi = wi.normalize();
			float GTerm = (max(Dot(wi, shadingData.sNormal), 0.0f) * max(-Dot(wi, light->normal(shadingData, wi)), 0.0f)) / l;
			if (GTerm > 0)
			{
				// Trace
				if (scene->visible(shadingData.x, p))
				{
					// Shade
					return shadingData.bsdf->evaluate(shadingData, wi) * emitted * GTerm / (pmf * pdf);
				}
			}
		}
		else
		{
			// Calculate GTerm
			Vec3 wi = p;
			float GTerm = max(Dot(wi, shadingData.sNormal), 0.0f);
			if (GTerm > 0)
			{
				// Trace
				if (scene->visible(shadingData.x, shadingData.x + (p * 10000.0f)))
				{
					// Shade
					return shadingData.bsdf->evaluate(shadingData, wi) * emitted * GTerm / (pmf * pdf);
				}
			}
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}

	const int MAX_DEPTH = 10;

	Colour pathTrace(Ray& r, Colour& pathThroughput, int depth, Sampler* sampler, bool canHitLight = true)
	{
		// Add pathtracer code here
		IntersectionData intersection = scene->bvh->traverse(r, scene->triangles);
		//IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				if (canHitLight == true)
				{
					return pathThroughput * shadingData.bsdf->emit(shadingData, shadingData.wo);
				}
				else
				{
					return Colour(0.0f, 0.0f, 0.0f);
				}
			}
			Colour direct = pathThroughput * computeDirect(shadingData, sampler);
			if (depth > MAX_DEPTH)
			{
				return direct;
			}
			float russianRouletteProbability = min(pathThroughput.Lum(), 0.9f);
			if (sampler->next() < russianRouletteProbability)
			{
				pathThroughput = pathThroughput / russianRouletteProbability;
			}
			else
			{
				return direct;
			}
			Colour bsdf;
			float pdf;
			Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
			pdf = SamplingDistributions::cosineHemispherePDF(wi);
			wi = shadingData.frame.toWorld(wi);
			bsdf = shadingData.bsdf->evaluate(shadingData, wi);
			pathThroughput = pathThroughput * bsdf * fabsf(Dot(wi, shadingData.sNormal)) / pdf;
			r.init(shadingData.x + (wi * EPSILON), wi);
			return (direct + pathTrace(r, pathThroughput, depth + 1, sampler, shadingData.bsdf->isPureSpecular()));
		}
		return scene->background->evaluate(shadingData, r.dir);
	}

	Colour direct(Ray& r, Sampler* sampler)
	{
		// Compute direct lighting for an image sampler here
		IntersectionData intersection = scene->bvh->traverse(r, scene->triangles);
		//IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				return shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return computeDirect(shadingData, sampler);
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}

	Colour albedo(Ray& r)
	{
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				return shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return shadingData.bsdf->evaluate(shadingData, Vec3(0, 1, 0));
		}
		return scene->background->evaluate(shadingData, r.dir);
	}
	Colour viewNormals(Ray& r)
	{
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t < FLT_MAX)
		{
			ShadingData shadingData = scene->calculateShadingData(intersection, r);
			return Colour(fabsf(shadingData.sNormal.x), fabsf(shadingData.sNormal.y), fabsf(shadingData.sNormal.z));
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}

	// Single Thread rendering
	void STrender() {
		for (unsigned int y = 0; y < film->height; y++)
		{
			for (unsigned int x = 0; x < film->width; x++)
			{
				float px = x + 0.5f;
				float py = y + 0.5f;
				Ray ray = scene->camera.generateRay(px, py);
				//Colour col = viewNormals(ray);
				//Colour col = direct(ray, samplers);
				Colour throughput(1.0f, 1.0f, 1.0f);  // init throughput
				Colour col = pathTrace(ray, throughput, 0, samplers);
				//Colour col = albedo(ray);
				film->splat(px, py, col);
				unsigned char r = (unsigned char)(col.r * 255);
				unsigned char g = (unsigned char)(col.g * 255);
				unsigned char b = (unsigned char)(col.b * 255);
				film->tonemap(x, y, r, g, b);
				canvas->draw(x, y, r, g, b);
			}
		}
	}
	// Tile based rendering
	void MTrender() {
		const int width = (int)film->width;
		const int height = (int)film->height;

		//for adaptive rendering
		const int maxSPP = 64;
		const float threshold = 0.001f;

		// tile size
		const int tileSize = 32;

		// count of tiles
		const int numTilesX = (width + tileSize - 1) / tileSize;
		const int numTilesY = (height + tileSize - 1) / tileSize;
		std::vector<Tile> tiles;

		for (int ty = 0; ty < numTilesY; ty++) {
			for (int tx = 0; tx < numTilesX; tx++) {
				int x0 = tx * tileSize;
				int y0 = ty * tileSize;
				int x1 = min(x0 + tileSize, width);
				int y1 = min(y0 + tileSize, height);
				tiles.emplace_back(x0, y0, x1, y1);
			}
		}

		// atomic counters to allocate tiles to threads
		std::atomic<int> nextTileIndex(0);

		auto worker = [&](int threadID) {
			Sampler* sampler = &samplers[threadID];
			while (true) {
				// get the tile index
				int tileIndex = nextTileIndex.fetch_add(1);
				if (tileIndex >= tiles.size()) break;
				Tile& tile = tiles[tileIndex];
				if (tile.converged) return; // if converged, sampling over

				for (int y = tile.y0; y < tile.y1; y++) {
					for (int x = tile.x0; x < tile.x1; x++) {
						// do the same thing with render()
						float px = x + 0.5f;
						float py = y + 0.5f;
						Ray ray = scene->camera.generateRay(px, py);

						Colour throughput(1.0f, 1.0f, 1.0f);  // init throughput
						Colour col = pathTrace(ray, throughput, 0, sampler);
						film->splat(px, py, col);

						// Welford update
						tile.spp++;
						Colour delta = col - tile.mean;
						tile.mean = tile.mean + delta * (1.0f / tile.spp);
						tile.M2 = tile.M2 + delta * (col - tile.mean);

						unsigned char r = (unsigned char)(col.r * 255);
						unsigned char g = (unsigned char)(col.g * 255);
						unsigned char b = (unsigned char)(col.b * 255);
						film->tonemap(x, y, r, g, b);
						canvas->draw(x, y, r, g, b);
					}
				}

				if (tile.spp >= 4) {
					Colour var = tile.M2 / (float)(tile.spp - 1);
					float lumVar = var.Lum();
					if (lumVar < threshold || tile.spp >= maxSPP) {
						tile.converged = true;
					}
				}
			}};

		// start threads
		std::vector<std::thread> threadPool;
		for (int i = 0; i < numProcs; i++)
		{
			threadPool.emplace_back(worker, i);
		}

		for (auto& t : threadPool)
		{
			t.join();
		}
	}
	void render()
	{
		film->incrementSPP();
		MTrender();
	}
	int getSPP()
	{
		return film->SPP;
	}
	void saveHDR(std::string filename)
	{
		film->save(filename);
	}
	void savePNG(std::string filename)
	{
		stbi_write_png(filename.c_str(), canvas->getWidth(), canvas->getHeight(), 3, canvas->getBackBuffer(), canvas->getWidth() * 3);
	}
};