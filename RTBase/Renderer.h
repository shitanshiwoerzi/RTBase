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
#include "Denoise.h"

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

	//Colour computeDirect(ShadingData shadingData, Sampler* sampler)
	//{
	//	if (shadingData.bsdf->isPureSpecular() == true)
	//	{
	//		return Colour(0.0f, 0.0f, 0.0f);
	//	}
	//	// Sample a light
	//	float pmf;
	//	Light* light = scene->sampleLight(sampler, pmf);
	//	// Sample a point on the light
	//	float pdf;
	//	Colour emitted;
	//	Vec3 p = light->sample(shadingData, sampler, emitted, pdf);
	//	if (light->isArea())
	//	{
	//		// Calculate GTerm
	//		Vec3 wi = p - shadingData.x;
	//		float l = wi.lengthSq();
	//		wi = wi.normalize();
	//		float GTerm = (max(Dot(wi, shadingData.sNormal), 0.0f) * max(-Dot(wi, light->normal(shadingData, wi)), 0.0f)) / l;
	//		if (GTerm > 0)
	//		{
	//			// Trace
	//			if (scene->visible(shadingData.x, p))
	//			{
	//				// Shade
	//				return shadingData.bsdf->evaluate(shadingData, wi) * emitted * GTerm / (pmf * pdf);
	//			}
	//		}
	//	}
	//	else
	//	{
	//		// Calculate GTerm
	//		Vec3 wi = p;
	//		float GTerm = max(Dot(wi, shadingData.sNormal), 0.0f);
	//		if (GTerm > 0)
	//		{
	//			// Trace
	//			if (scene->visible(shadingData.x, shadingData.x + (p * 10000.0f)))
	//			{
	//				// Shade
	//				return shadingData.bsdf->evaluate(shadingData, wi) * emitted * GTerm / (pmf * pdf);
	//			}
	//		}
	//	}
	//	return Colour(0.0f, 0.0f, 0.0f);
	//}

	Colour computeDirect(const ShadingData& shadingData, Sampler* sampler)
	{
		if (shadingData.bsdf->isPureSpecular())
			return Colour(0.0f, 0.0f, 0.0f);

		Colour result(0.0f, 0.0f, 0.0f);

		//----------------------------
		// (A) Light Sampling
		//----------------------------
		//   - 在场景光源列表里抽取一个 Light，概率 pmf
		float pmf;
		Light* light = scene->sampleLight(sampler, pmf);
		if (light)
		{
			float lightPdf;
			Colour Le;
			// 从光源上采样一个方向 / 位置
			Vec3 wi = light->sample(shadingData, sampler, Le, lightPdf);

			if (lightPdf > 1e-6f && pmf > 1e-6f && Le.Lum() > 0.0f)
			{
				// 可见性判断
				bool visible = false;

				// 如果是 AreaLight: p是坐标 => GTerm；如果是 EnvLight: p是方向 => cosTerm
				if (light->isArea())
				{
					// area light case
					// p = shadingData.x + direction*distance
					// light->normal(...) => 计算三角形法线
					// GTerm = cos(θx)*cos(θl)/r²
					// 先判断可见性
					Vec3 toLight = wi - shadingData.x; // 这里wi是位置
					float dist2 = toLight.lengthSq();
					Vec3 dir = toLight.normalize();
					if (scene->visible(shadingData.x, wi))
					{
						float cos_x = max(Dot(dir, shadingData.sNormal), 0.0f);
						float cos_l = max(-Dot(dir, light->normal(shadingData, dir)), 0.0f);
						float G = (cos_x * cos_l) / dist2;
						// evaluate BSDF
						Colour f = shadingData.bsdf->evaluate(shadingData, dir);

						// 还要 BSDF PDF => 这里可选 => MIS
						float bsdfPdf = shadingData.bsdf->PDF(shadingData, dir);

						// MIS weight
						float w_light = (lightPdf * pmf) * (lightPdf * pmf);
						float w_bsdf = bsdfPdf * bsdfPdf; // or skip MIS for area
						float weight = w_light / (w_light + w_bsdf + 1e-6f);

						result = result + f * Le * G * weight / (lightPdf * pmf);
					}
				}
				else
				{
					// environment light case
					// wi就是一个方向
					if (scene->visible(shadingData.x, shadingData.x + wi * 1e4f))
					{
						// BSDF evaluate
						Colour f = shadingData.bsdf->evaluate(shadingData, wi);
						float cosTerm = max(Dot(wi, shadingData.sNormal), 0.0f);

						// BSDF PDF
						float bsdfPdf = shadingData.bsdf->PDF(shadingData, wi);

						// MIS
						float w_light = (lightPdf * pmf) * (lightPdf * pmf);
						float w_bsdf = bsdfPdf * bsdfPdf;
						float weight = w_light / (w_light + w_bsdf + 1e-6f);

						result = result + f * Le * cosTerm * weight / (lightPdf * pmf);
					}
				}
			}
		}

		//----------------------------
		// (B) BSDF Sampling
		//----------------------------
		float bsdfPdf;
		Colour bsdfVal;
		Vec3 wi_bsdf = shadingData.bsdf->sample(shadingData, sampler, bsdfVal, bsdfPdf);

		if (bsdfPdf > 1e-6f && bsdfVal.Lum() > 0.0f)
		{
			float cosTerm = max(Dot(wi_bsdf, shadingData.sNormal), 0.0f);
			if (cosTerm > 0.0f)
			{
				// shadow ray
				if (scene->visible(shadingData.x, shadingData.x + wi_bsdf * 1e4f))
				{
					// check if hit area light or environment
					Colour Le(0.0f, 0.0f, 0.0f);
					float lightPdf = 0.0f;
					float pmfLocal = 1.0f; // assume single env or we do logic if multiple lights

					// environment?
					if (scene->background)
					{
						Le = scene->background->evaluate(shadingData, wi_bsdf);
						lightPdf = scene->background->PDF(shadingData, wi_bsdf);
					}

					// 也可以做 if we hit triangle with isLight()
					// → IntersectionData check

					if (lightPdf > 1e-6f && Le.Lum() > 0.0f)
					{
						// MIS
						float w_bsdf = bsdfPdf * bsdfPdf;
						float w_light = (lightPdf * pmfLocal) * (lightPdf * pmfLocal);
						float weight = w_bsdf / (w_bsdf + w_light + 1e-6f);

						result = result + bsdfVal * Le * cosTerm * weight / bsdfPdf;
					}
				}
			}
		}

		return result;
	}

	const int MAX_DEPTH = 10;

	Colour pathTrace(Ray& r, Colour& pathThroughput, int depth, Sampler* sampler, bool canHitLight = true)
	{
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

			Colour indirect;
			float pdf;
			Vec3 wi = shadingData.bsdf->sample(shadingData, sampler, indirect, pdf);
			if (pdf < 1e-6f || !std::isfinite(pdf) || indirect.hasNaN())
			{
				return direct;
			}
			float cosTerm = fmaxf(0.0f, Dot(wi, shadingData.sNormal));// Cosine term

			if (!std::isfinite(cosTerm) || !std::isfinite(indirect.r) || !std::isfinite(indirect.g) || !std::isfinite(indirect.b))
			{
				return direct;
			}

			pathThroughput = pathThroughput * indirect * (cosTerm / pdf);// accumulate
			r.init(shadingData.x + (wi * EPSILON), wi);
			return (direct + pathTrace(r, pathThroughput, depth + 1, sampler, shadingData.bsdf->isPureSpecular()));
		}
		return scene->background->evaluate(shadingData, r.dir);
	}

	// Compute direct lighting for an image sampler
	Colour direct(Ray& r, Sampler* sampler)
	{
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
		//IntersectionData intersection = scene->traverse(r);
		IntersectionData intersection = scene->bvh->traverse(r, scene->triangles);
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
		//IntersectionData intersection = scene->traverse(r);
		IntersectionData intersection = scene->bvh->traverse(r, scene->triangles);
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

						if (tile.spp == 0) {
							int idx = y * width + x;
							film->albedoBuffer[idx] = albedo(ray);
							film->normalBuffer[idx] = viewNormals(ray);
						}

						Colour throughput(1.0f, 1.0f, 1.0f);  // init throughput
						Colour col = pathTrace(ray, throughput, 0, sampler);
						film->splat(px, py, col);

						// Welford update
						tile.spp++;
						Colour delta = col - tile.mean;
						tile.mean = tile.mean + delta * (1.0f / tile.spp);
						tile.M2 = tile.M2 + delta * (col - tile.mean);

						// tonemap + draw
						unsigned char r = (unsigned char)(col.r * 255);
						unsigned char g = (unsigned char)(col.g * 255);
						unsigned char b = (unsigned char)(col.b * 255);
						film->tonemap(x, y, r, g, b);
						canvas->draw(x, y, r, g, b);
					}
				}

				// check variance
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
		float* denoised = new float[film->width * film->height * 3];
		denoiseWithOIDN(reinterpret_cast<float*>(film->film),
			reinterpret_cast<float*>(film->albedoBuffer),
			reinterpret_cast<float*>(film->normalBuffer),
			denoised,
			film->width, film->height);

		stbi_write_hdr("raw.hdr", film->width, film->height, 3, (float*)film->film);
		stbi_write_hdr("denoised.hdr", film->width, film->height, 3, denoised);
		saveHDRTonemapped("denoised_tonemapped.hdr", denoised, film->width, film->height, film->SPP, 0.5f); // try lower exposure if still too bright
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