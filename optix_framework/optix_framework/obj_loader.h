
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
#include <sutil.h>
#include <optix_world.h>
#include <glm.h>
#include <string>
#include <folders.h>
#include <area_light.h>
#include "shader.h"
#include "mesh.h"
#include <memory>

class MaterialHost;

//-----------------------------------------------------------------------------
// 
//  ObjLoader class declaration 
//
//-----------------------------------------------------------------------------
class ObjLoader
{
public:
  ObjLoader( const char* filename,                 // Model filename
                      optix::Context& context,               // Context for RT object creation
                      optix::GeometryGroup geometrygroup,   // Empty geom group to hold model
                      const char* ASBuilder   = "Sbvh",
                      const char* ASTraverser = "Bvh",
                      const char* ASRefine = "0",
                      bool large_geom = false ); 
  
  // Reorder geom data for page fault minimization
  ObjLoader( const char* filename,
                      optix::Context& context,
                      optix::GeometryGroup geometrygroup,
                      optix::Material material,                // Material override
                      bool force_load_material_params = false, // Set obj_material params even though material is overridden
                      const char* ASBuilder   = "Sbvh",
                      const char* ASTraverser = "Bvh",
                      const char* ASRefine = "0",
                      bool large_geom = false);                // Reorder geom data for page fault minimization

    virtual ~ObjLoader() {} // makes sure CRT objects are destroyed on the correct heap

  virtual void setIntersectProgram( optix::Program program );
  virtual void setBboxProgram(optix::Program program);
  virtual std::vector<Mesh> load();
  virtual std::vector<Mesh> load(const optix::Matrix4x4& transform);

  virtual optix::Aabb getSceneBBox()const { return m_aabb; }

  virtual void getAreaLights(std::vector<TriangleLight> & lights);
  static bool isMyFile(const char* filename);

  std::vector<std::shared_ptr<MaterialHost>> getMaterialParameters()
  {
	  return m_material_params;
  }

protected:

  void createMaterial();
  std::vector<Mesh> createGeometryInstances(GLMmodel* model);
  void loadVertexData( GLMmodel* model, const optix::Matrix4x4& transform );
  void createMaterialParams( GLMmodel* model );
  std::shared_ptr<MaterialHost> getMaterial(unsigned int index);

  void createLightBuffer( GLMmodel* model, const optix::Matrix4x4& transform );

  void extractAreaLights(const GLMmodel * model, const optix::Matrix4x4 & transform, unsigned int & num_light, std::vector<TriangleLight> & lights);

  std::vector<TriangleLight> m_lights;
  std::string            m_pathname;
  std::string            m_filename;
  optix::Context         m_context;
  optix::GeometryGroup   m_geometrygroup;
  optix::Buffer          m_vbuffer;
  optix::Buffer          m_nbuffer;
  optix::Buffer          m_tbuffer;
  optix::Material        m_material;

  optix::Buffer          m_light_buffer;
  bool                   m_have_default_material;
  bool                   m_force_load_material_params;
  const char*            m_ASBuilder;
  const char*            m_ASTraverser;
  const char*            m_ASRefine;
  bool                   m_large_geom;
  optix::Aabb            m_aabb;
  std::vector<std::shared_ptr<MaterialHost>> m_material_params;
};



