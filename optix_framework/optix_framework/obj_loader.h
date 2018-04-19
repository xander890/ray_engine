
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#pragma once

#include <optix_world.h>
#include <glm.h>
#include <string>
#include <folders.h>
#include <area_light.h>
#include "shader.h"
#include "mesh.h"
#include <memory>

class MaterialHost;

struct ObjMaterial
{
	std::string name = "empty";        /* name of material */
	int diffuse_tex = -1;       // Kd diffuse component
	int ambient_tex = -1;       // Ka ambient component
	int specular_tex = -1;      // Ks specular component
	optix::float3 emissive = optix::make_float3(0);      // emissive component
	optix::float3 absorption = optix::make_float3(0);;   // Added: absorption.
	optix::float3 scattering = optix::make_float3(0);;   // Added: absorption.
	optix::float3 asymmetry = optix::make_float3(0);;   // Added: absorption.
	float scale = 1;   // Added: absorption.
	float shininess = 1;        // Ns specular exponent
	float refraction = 0;       // Tr
	float ior = 1;
	float alpha = 1;            // d
	float reflectivity = 1;     // reflection
	int   illum = 0;           // illum
};

//-----------------------------------------------------------------------------
// 
//  ObjLoader class declaration 
//
//-----------------------------------------------------------------------------
class ObjLoader
{
public:
  ObjLoader( const char* filename,                 // Model filename
                      optix::Context& context); 
  
  virtual ~ObjLoader() {} // makes sure CRT objects are destroyed on the correct heap
	
  virtual void setIntersectProgram( optix::Program program );
  virtual void setBboxProgram(optix::Program program);
  virtual std::vector<std::unique_ptr<Object>> load();
  virtual std::vector<std::unique_ptr<Object>> load(const optix::Matrix4x4& transform);

  virtual optix::Aabb getSceneBBox()const { return m_aabb; }

  virtual void getAreaLights(std::vector<TriangleLight> & lights);
  static bool isMyFile(const char* filename);

  std::vector<std::shared_ptr<MaterialHost>> getMaterialParameters()
  {
	  return m_material_params;
  }

  static std::vector<ObjMaterial> parse_mtl_file(std::string mtl, optix::Context & ctx);
  static ObjMaterial convert_mat(GLMmaterial& mat, optix::Context ctx);

protected:

  std::vector<std::unique_ptr<Object>> createGeometryInstances(GLMmodel* model);
  void loadVertexData( GLMmodel* model, const optix::Matrix4x4& transform );
  void createMaterialParams( GLMmodel* model );
  std::shared_ptr<MaterialHost> getMaterial(unsigned int index);

  void createLightBuffer( GLMmodel* model, const optix::Matrix4x4& transform );

  void extractAreaLights(const GLMmodel * model, const optix::Matrix4x4 & transform, unsigned int & num_light, std::vector<TriangleLight> & lights);

  std::vector<TriangleLight> m_lights;
  std::string            m_pathname;
  std::string            m_filename;
  optix::Context         m_context;
  optix::Buffer          m_vbuffer;
  optix::Buffer          m_nbuffer;
  optix::Buffer          m_tbuffer;

  optix::Buffer          m_light_buffer;
  optix::Aabb            m_aabb;
  std::vector<std::shared_ptr<MaterialHost>> m_material_params;

};


