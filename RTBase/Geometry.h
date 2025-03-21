#pragma once

#include "Core.h"
#include "Sampling.h"

class Ray
{
public:
	Vec3 o;
	Vec3 dir;
	Vec3 invDir;
	Ray()
	{
	}
	Ray(Vec3 _o, Vec3 _d)
	{
		init(_o, _d);
	}
	void init(Vec3 _o, Vec3 _d)
	{
		o = _o;
		dir = _d;
		invDir = Vec3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
	}
	Vec3 at(const float t) const
	{
		return (o + (dir * t));
	}
};

class Plane
{
public:
	Vec3 n;
	float d;
	void init(Vec3& _n, float _d)
	{
		n = _n;
		d = _d;
	}
	// Add code here
	bool rayIntersect(Ray& r, float& t)
	{
		t = -(n.dot(r.o) + d) / (n.dot(r.dir));
		return t >= 0;
	}
};

#define EPSILON 0.001f

class Triangle
{
public:
	Vertex vertices[3];
	Vec3 e0; //Edge 0
	Vec3 e1; // Edge 1
	Vec3 e2; // Edge 2
	Vec3 n; // Geometric Normal
	float area; // Triangle area
	float d; // For ray triangle if needed
	unsigned int materialIndex;
	void init(Vertex v0, Vertex v1, Vertex v2, unsigned int _materialIndex)
	{
		materialIndex = _materialIndex;
		vertices[0] = v0;
		vertices[1] = v1;
		vertices[2] = v2;
		e0 = vertices[1].p - vertices[0].p;
		e1 = vertices[2].p - vertices[1].p;
		e2 = vertices[0].p - vertices[2].p;
		n = e1.cross(e2).normalize();
		area = e1.cross(e2).length() * 0.5f;
		d = Dot(n, vertices[0].p);
	}
	Vec3 centre() const
	{
		return (vertices[0].p + vertices[1].p + vertices[2].p) / 3.0f;
	}

	// Add code here
	//bool rayIntersect(const Ray& r, float& t, float& u, float& v) const
	//{
	//	t = -(n.dot(r.o) + d) / (n.dot(r.dir));
	//	if (t < 0) return false;
	//	Vec3 p = r.at(t);
	//	Vec3 q1 = p - vertices[1].p;
	//	Vec3 q2 = p - vertices[2].p;
	//	float C1 = Dot(e1.cross(q1), n);
	//	float C2 = Dot(e2.cross(q2), n);
	//	u = C1 / area;
	//	if (u < 0 || u > 1.0f) { return false; }
	//	v = C2 / area;
	//	if (v < 0 || (u + v) > 1.0f) { return false; }
	//	return true;
	//}

	bool rayIntersect(const Ray& r, float& t, float& u, float& v) const {
		float denom = Dot(n, r.dir);
		if (denom == 0) { return false; }
		t = (d - Dot(n, r.o)) / denom;
		if (t < 0) { return false; }
		Vec3 p = r.at(t);
		float invArea = 1.0f / Dot(e1.cross(e2), n);
		u = Dot(e1.cross(p - vertices[1].p), n) * invArea;
		if (u < 0 || u > 1.0f)  return false;
		v = Dot(e2.cross(p - vertices[2].p), n) * invArea;
		if (v < 0 || (u + v) > 1.0f)  return false;
		return true;
	}

	bool Moller_RayIntersect(const Ray& r, float& t, float& u, float& v) const {

		Vec3 de1 = vertices[1].p - vertices[0].p;
		Vec3 de2 = vertices[2].p - vertices[0].p;

		Vec3 pvec = r.dir.cross(de2);
		float det = de1.dot(pvec);
		if (std::fabs(det) < 1e-8f) return false;
		float invdet = 1.0f / det;
		Vec3 T = r.o - vertices[0].p;
		u = T.dot(pvec) * invdet;
		if (u < 0 || u > 1.0f) return false;
		Vec3 qvec = T.cross(de1);
		v = r.dir.dot(qvec) * invdet;
		if (v < 0 || (u + v) > 1.0f) return false;
		t = de2.dot(qvec) * invdet;
		return t > 1e-8f;
	}

	void interpolateAttributes(const float alpha, const float beta, const float gamma, Vec3& interpolatedNormal, float& interpolatedU, float& interpolatedV) const
	{
		interpolatedNormal = vertices[0].normal * alpha + vertices[1].normal * beta + vertices[2].normal * gamma;
		interpolatedNormal = interpolatedNormal.normalize();
		interpolatedU = vertices[0].u * alpha + vertices[1].u * beta + vertices[2].u * gamma;
		interpolatedV = vertices[0].v * alpha + vertices[1].v * beta + vertices[2].v * gamma;
	}
	// Add code here
	Vec3 sample(Sampler* sampler, float& pdf)
	{
		float r1 = sampler->next();  // 生成随机数 r1
		float r2 = sampler->next();  // 生成随机数 r2

		// 将 (r1, r2) 转换为三角形上的点
		float sqrtR1 = sqrt(r1);
		float u = 1.0f - sqrtR1;
		float v = r2 * sqrtR1;

		Vec3 sampledPoint = vertices[0].p * u + vertices[1].p * v + vertices[2].p * (1 - u - v); // 重心坐标变换
		pdf = 1.0f / area;  // 计算 PDF (均匀分布)
		return sampledPoint;
	}
	Vec3 gNormal()
	{
		return (n * (Dot(vertices[0].normal, n) > 0 ? 1.0f : -1.0f));
	}
};

class AABB
{
public:
	Vec3 max;
	Vec3 min;
	AABB()
	{
		reset();
	}
	void reset()
	{
		max = Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		min = Vec3(FLT_MAX, FLT_MAX, FLT_MAX);
	}
	void extend(const Vec3 p)
	{
		max = Max(max, p);
		min = Min(min, p);
	}
	// Add code here
	bool rayAABB(const Ray& r, float& t)
	{
		float tmin = std::numeric_limits<float>::infinity();
		float tmax = std::numeric_limits<float>::infinity();

		for (int i = 0; i < 3; i++) {
			float origin = (i == 0) ? r.o.x : (i == 1) ? r.o.y : r.o.z;
			float dir = (i == 0) ? r.dir.x : (i == 1) ? r.dir.y : r.dir.z;
			float minVal = (i == 0) ? min.x : (i == 1) ? min.y : min.z;
			float maxVal = (i == 0) ? max.x : (i == 1) ? max.y : max.z;

			if (std::fabs(dir) < 1e-6f) { // parallel the axis
				if (origin < minVal || origin > maxVal)
					return false; // outside the box
			}
			else {
				float t0 = (minVal - origin) / dir;
				float t1 = (maxVal - origin) / dir;
				if (t0 > t1) std::swap(t0, t1); // ensure t0 is min
				tmin = std::max(tmin, t0);
				tmax = std::min(tmax, t1);

				if (tmin > tmax) return false; // no intersection
			}
		}

		if (tmax < 0) return false;  // AABB is in the inverse ray direction
		t = tmin;
		return true;
	}

	// Add code here
	bool rayAABB(const Ray& r)
	{
		float t;
		return rayAABB(r, t);
	}
	// Add code here
	float area()
	{
		Vec3 size = max - min;
		return ((size.x * size.y) + (size.y * size.z) + (size.x * size.z)) * 2.0f;
	}
};

class Sphere
{
public:
	Vec3 centre;
	float radius;
	void init(Vec3& _centre, float _radius)
	{
		centre = _centre;
		radius = _radius;
	}
	// Add code here
	bool rayIntersect(Ray& r, float& t)
	{
		Vec3 l = r.o - centre;
		float b = l.dot(r.dir);
		float c = l.dot(l) - radius * radius;
		float dis = b * b - c;
		if (dis < 0) return false;

		float sqrtD = std::sqrt(dis);
		float t1 = -b - sqrtD;
		float t2 = -b + sqrtD;
		if (t1 > 0) {
			t = t1;
			return true;
		}
		if (t2 > 0) {
			t = t2;
			return true;
		}
		return false;
	}
};

struct IntersectionData
{
	unsigned int ID;
	float t;
	float alpha;
	float beta;
	float gamma;
};

#define MAXNODE_TRIANGLES 8
#define TRAVERSE_COST 1.0f
#define TRIANGLE_COST 2.0f
#define BUILD_BINS 32

class BVHNode
{
public:
	AABB bounds;
	BVHNode* r;
	BVHNode* l;

	// This can store an offset and number of triangles in a global triangle list for example
	// But you can store this however you want!
	 unsigned int offset;
	 unsigned char num;
	BVHNode()
	{
		r = NULL;
		l = NULL;
	}
	// Note there are several options for how to implement the build method. Update this as required
	void build(std::vector<Triangle>& inputTriangles)
	{
		// Add BVH building code here
		
	}
	void traverse(const Ray& ray, const std::vector<Triangle>& triangles, IntersectionData& intersection)
	{
		// Add BVH Traversal code here
	
	}

	IntersectionData traverse(const Ray& ray, const std::vector<Triangle>& triangles)
	{
		IntersectionData intersection;
		intersection.t = FLT_MAX;
		traverse(ray, triangles, intersection);
		return intersection;
	}
	bool traverseVisible(const Ray& ray, const std::vector<Triangle>& triangles, const float maxT)
	{
		// Add visibility code here
		for (int i = 0; i < triangles.size(); i++)
		{
			float t;
			float u;
			float v;
			if (triangles[i].rayIntersect(ray, t, u, v))
			{
				if (t < maxT)
				{
					return false;
				}
			}
		}
		return true;
	}
};
