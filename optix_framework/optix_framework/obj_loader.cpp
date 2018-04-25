
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

#include <obj_loader.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <string.h>
#include <optixu/optixu.h>
#include <ImageLoader.h>
#include <area_light.h>

#include "object_host.h"
#include "host_material.h"
#include <scattering_material.h>

//------------------------------------------------------------------------------
// 
//  Helper functions
//
//------------------------------------------------------------------------------
using optix::Buffer;
using optix::int3;
using optix::float3;
using optix::float4;
using optix::make_float3;

namespace 
{
  std::string getExtension( const std::string& filename )
  {
	// Get the filename extension
	std::string::size_type extension_index = filename.find_last_of( "." );
	return extension_index != std::string::npos ?
		   filename.substr( extension_index+1 ) :
		   std::string();
  }
}

//------------------------------------------------------------------------------
// 
//  ObjLoader class definition 
//
//------------------------------------------------------------------------------

ObjLoader::ObjLoader( const char* filename,
					  optix::Context& context)
: m_filename( filename ),
  m_context( context ),
  m_vbuffer( 0 ),
  m_nbuffer( 0 ),
  m_tbuffer( 0 ),
  m_aabb(),
  m_lights()
{
  m_pathname = m_filename.substr(0,m_filename.find_last_of("/\\")+1);
}

std::vector<std::unique_ptr<Object>> ObjLoader::load()
{
   return load( optix::Matrix4x4::identity() );
}

void ObjLoader::setIntersectProgram( optix::Program intersect_program )
{
 // m_intersect_program = intersect_program;
}

void ObjLoader::setBboxProgram( optix::Program bbox_program )
{
 // m_bbox_program = bbox_program;
}

std::vector<std::unique_ptr<Object>> ObjLoader::load(const optix::Matrix4x4& transform)
{
  // parse the OBJ file
	std::string s = m_filename;
	GLMmodel* model = glmReadOBJ(s.c_str());
  if ( !model ) {
	std::stringstream ss;
	ss << "ObjLoader::loadImpl - glmReadOBJ( '" << m_filename << "' ) failed" << std::endl;
	throw optix::Exception( ss.str() );
  }
  
  // Create vertex data buffers to be shared by all Geometries
  loadVertexData( model, transform );

  // Create a GeometryInstance and Geometry for each obj group
  createMaterialParams( model );
  auto meshes = createGeometryInstances( model );
  
  // Create a data for sampling light sources
  createLightBuffer( model, transform );

  glmDelete( model );
  return meshes;
}


void ObjLoader::loadVertexData( GLMmodel* model, const optix::Matrix4x4& transform )
{
  unsigned int num_vertices  = model->numvertices;
  unsigned int num_texcoords = model->numtexcoords;
  unsigned int num_normals   = model->numnormals;

  // Create vertex buffer
  m_vbuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_vertices );
  float3* vbuffer_data = static_cast<float3*>( m_vbuffer->map() );

  // Create normal buffer
  m_nbuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_normals );
  float3* nbuffer_data = static_cast<float3*>( m_nbuffer->map() );

  // Create texcoord buffer
  m_tbuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, num_texcoords );
	optix::float2* tbuffer_data = static_cast<optix::float2*>( m_tbuffer->map() );

  // Transform and copy vertices.  
  for ( unsigned int i = 0; i < num_vertices; ++i )
  {
	const float3 v3 = *reinterpret_cast<float3*>(&model->vertices[(i+1)*3]);
	  optix::float4 v4 = make_float4( v3, 1.0f );
	vbuffer_data[i] = optix::make_float3( transform*v4 );
  }

  // Transform and copy normals.
  const optix::Matrix4x4 norm_transform = transform.inverse().transpose();
  for( unsigned int i = 0; i < num_normals; ++i )
  {
	const float3 v3 = *reinterpret_cast<float3*>(&model->normals[(i+1)*3]);
	  optix::float4 v4 = make_float4( v3, 0.0f );
	nbuffer_data[i] = optix::make_float3( norm_transform*v4 );
  }

  // Copy texture coordinates.
  memcpy( static_cast<void*>( tbuffer_data ),
		  static_cast<void*>( &(model->texcoords[2]) ),
		  sizeof( float )*num_texcoords*2 );   

  // Calculate bbox of model
  for( unsigned int i = 0; i < num_vertices; ++i )
	m_aabb.include( vbuffer_data[i] );

  // Unmap buffers.
  m_vbuffer->unmap();
  m_nbuffer->unmap();
  m_tbuffer->unmap();
}


std::vector<ObjMaterial> ObjLoader::parse_mtl_file(std::string mat, optix::Context & ctx)
{
	GLMmodel* model = new GLMmodel();
	std::vector<char> writable(mat.begin(), mat.end());
	writable.push_back('\0');
	model->pathname = "";
	model->mtllibname = NULL;
	model->numvertices = 0;
	model->vertexColors = NULL;
	model->numnormals = 0;
	model->normals = NULL;
	model->numtexcoords = 0;
	model->texcoords = NULL;
	model->numfacetnorms = 0;
	model->facetnorms = NULL;
	model->numtriangles = 0;
	model->triangles = NULL;
	model->nummaterials = 0;
	model->materials = NULL;
	model->numgroups = 0;
	model->groups = NULL;
	model->position[0] = 0.0;
	model->position[1] = 0.0;
	model->position[2] = 0.0;
	model->usePerVertexColors = 0;
	_glmReadMTL(model, &writable[0]);
	std::vector<ObjMaterial> vec;
	for (unsigned int i = 0; i < model->nummaterials; i++)
		vec.push_back(convert_mat(*model->materials, ctx));
	return vec;
}

std::vector<std::unique_ptr<Object>> ObjLoader::createGeometryInstances(GLMmodel* model)
{
  std::vector<std::unique_ptr<Object>> instances;

  // Loop over all groups -- grab the triangles and material props from each group
  unsigned int triangle_count = 0u;
  unsigned int group_count = 0u;
  for ( GLMgroup* obj_group = model->groups;
		obj_group != 0;
		obj_group = obj_group->next, group_count++ ) {

	unsigned int num_triangles = obj_group->numtriangles;
	if ( num_triangles == 0 ) continue; 

	// Create vertex index buffers
	Buffer vindex_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, num_triangles );
	int3* vindex_buffer_data = static_cast<int3*>( vindex_buffer->map() );

	Buffer tindex_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, num_triangles );
	int3* tindex_buffer_data = static_cast<int3*>( tindex_buffer->map() );

	Buffer nindex_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, num_triangles );
	int3* nindex_buffer_data = static_cast<int3*>( nindex_buffer->map() );

	Buffer mbuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, num_triangles );
	unsigned int* mbuffer_data = static_cast<unsigned int*>( mbuffer->map() );

    optix::Aabb bbox;

	for ( unsigned int i = 0; i < obj_group->numtriangles; ++i, ++triangle_count ) {

	  unsigned int tindex = obj_group->triangles[i];
	  int3 vindices;
	  vindices.x = model->triangles[ tindex ].vindices[0] - 1; 
	  vindices.y = model->triangles[ tindex ].vindices[1] - 1; 
	  vindices.z = model->triangles[ tindex ].vindices[2] - 1; 
	  assert( vindices.x <= static_cast<int>(model->numvertices) );
	  assert( vindices.y <= static_cast<int>(model->numvertices) );
	  assert( vindices.z <= static_cast<int>(model->numvertices) );
	  
	  vindex_buffer_data[ i ] = vindices;

      if (vindices.x > 2 && vindices.y > 2 && vindices.z > 2)
      {
          float3 vx = *((float3*)&model->vertices[vindices.x * 3]); bbox.include(vx);
          float3 vy = *((float3*)&model->vertices[vindices.y * 3]); bbox.include(vy);
          float3 vz = *((float3*)&model->vertices[vindices.z * 3]); bbox.include(vz);
      }

	  int3 nindices;
	  nindices.x = model->triangles[ tindex ].nindices[0] - 1; 
	  nindices.y = model->triangles[ tindex ].nindices[1] - 1; 
	  nindices.z = model->triangles[ tindex ].nindices[2] - 1; 
	  assert( nindices.x <= static_cast<int>(model->numnormals) );
	  assert( nindices.y <= static_cast<int>(model->numnormals) );
	  assert( nindices.z <= static_cast<int>(model->numnormals) );

	  int3 tindices;
	  tindices.x = model->triangles[ tindex ].tindices[0] - 1; 
	  tindices.y = model->triangles[ tindex ].tindices[1] - 1; 
	  tindices.z = model->triangles[ tindex ].tindices[2] - 1; 
	  assert( tindices.x <= static_cast<int>(model->numtexcoords) );
	  assert( tindices.y <= static_cast<int>(model->numtexcoords) );
	  assert( tindices.z <= static_cast<int>(model->numtexcoords) );

	  nindex_buffer_data[ i ] = nindices;
	  tindex_buffer_data[ i ] = tindices;
	  mbuffer_data[ i ] = 0; // See above TODO

	}
	vindex_buffer->unmap();
	tindex_buffer->unmap();
	nindex_buffer->unmap();
	mbuffer->unmap();


    std::shared_ptr<MaterialHost> materialData = getMaterial(obj_group->material);
    MeshData meshdata = { m_vbuffer, m_nbuffer, m_tbuffer, vindex_buffer, nindex_buffer, tindex_buffer, (int)num_triangles, bbox };
    std::unique_ptr<Object> rtMesh = std::make_unique<Object>(m_context);
    std::string name = obj_group->name;
    std::unique_ptr<Geometry> geom = std::make_unique<Geometry>(m_context);
    geom->init(name.c_str(), meshdata);
    rtMesh->init(name.c_str(), std::move(geom), materialData);
    
	instances.push_back( std::move(rtMesh) );
  }
    return instances;
}

bool ObjLoader::isMyFile( const char* filename )
{
  return getExtension( filename ) == "obj";
}


std::shared_ptr<MaterialHost> ObjLoader::getMaterial(unsigned int index)
{
  // Load params from this material into the GI 
  if ( index < m_material_params.size() ) {
    return m_material_params[index];
  }
  return nullptr;
}

ObjMaterial ObjLoader::convert_mat(GLMmaterial& mat, optix::Context ctx)
{
	ObjMaterial params;

	params.shininess = mat.shininess;
	params.illum = mat.shader;
	params.alpha = mat.alpha;
	params.ior = mat.ior;
	params.name = std::string(mat.name);
	params.reflectivity = mat.reflectivity;
	params.refraction = mat.refraction;
	params.emissive = make_float3(mat.emissive[0], mat.emissive[1], mat.emissive[2]);
	params.absorption = make_float3(mat.absorption[0], mat.absorption[1], mat.absorption[2]);
	params.asymmetry = make_float3(mat.asymmetry[0], mat.asymmetry[1], mat.asymmetry[2]);
	params.scattering = make_float3(mat.scattering[0], mat.scattering[1], mat.scattering[2]);
	params.scale = mat.scale;

	float3 Kd = make_float3(mat.diffuse[0],
		mat.diffuse[1],
		mat.diffuse[2]);
	float3 Ka = make_float3(mat.ambient[0],
		mat.ambient[1],
		mat.ambient[2]);
	float3 Ks = make_float3(mat.specular[0],
		mat.specular[1],
		mat.specular[2]);

	// load textures relatively to OBJ main file
	std::string ambient_map = strlen(mat.ambient_map) ? Folders::texture_folder + mat.ambient_map : "";
	std::string diffuse_map = strlen(mat.diffuse_map) ? Folders::texture_folder + mat.diffuse_map : "";
	std::string specular_map = strlen(mat.specular_map) ? Folders::texture_folder + mat.specular_map : "";

	params.ambient_tex = std::move(loadTexture(ctx, ambient_map, Ka));
	params.diffuse_tex = std::move(loadTexture(ctx, diffuse_map, Kd));
	params.specular_tex = std::move(loadTexture(ctx, specular_map, Ks));
	return params;
}

void ObjLoader::createMaterialParams( GLMmodel* model )
{
  m_material_params.resize( model->nummaterials );
  for ( unsigned int i = 0; i < model->nummaterials; ++i ) {

	GLMmaterial& mat = model->materials[i];
	ObjMaterial params = convert_mat(mat, m_context);
    m_material_params[i] = std::make_shared<MaterialHost>(m_context, params);
  }
}

void ObjLoader::extractAreaLights(const GLMmodel * model, const optix::Matrix4x4 & transform, unsigned int & num_light, std::vector<TriangleLight> & lights)
{
	unsigned int group_count = 0u;

	if (model->nummaterials > 0)
	{
		for ( GLMgroup* obj_group = model->groups; obj_group != 0; obj_group = obj_group->next, group_count++ ) 
		{
			unsigned int num_triangles = obj_group->numtriangles;
			if ( num_triangles == 0 ) continue; 
			GLMmaterial& mat = model->materials[obj_group->material];

			if ( (mat.ambient[0] + mat.ambient[1] + mat.ambient[2]) > 0.0f ) 
			{
				// extract necessary data
				for ( unsigned int i = 0; i < obj_group->numtriangles; ++i ) 
				{
					// indices for vertex data
					unsigned int tindex = obj_group->triangles[i];
					int3 vindices;
					vindices.x = model->triangles[ tindex ].vindices[0]; 
					vindices.y = model->triangles[ tindex ].vindices[1]; 
					vindices.z = model->triangles[ tindex ].vindices[2]; 

					TriangleLight light;
					light.v1 = *((float3*)&model->vertices[vindices.x * 3]);
					float4 v = make_float4( light.v1, 1.0f );
					light.v1 = make_float3( transform*v);

					light.v2 = *((float3*)&model->vertices[vindices.y * 3]);
					v = make_float4( light.v2, 1.0f );
					light.v2 = make_float3( transform*v);

					light.v3 = *((float3*)&model->vertices[vindices.z * 3]);
					v = make_float4( light.v3, 1.0f );
					light.v3 = make_float3( transform*v);

					float3 area_vec = cross(light.v2 - light.v1, light.v3 - light.v1);
					light.area = 0.5f* length(area_vec);

					// normal vector
					light.normal = normalize(area_vec);

					light.emission = make_float3( mat.ambient[0], mat.ambient[1], mat.ambient[2] );

					lights.push_back(light);

					num_light++;
				}
			}
		}
	}
}

void ObjLoader::createLightBuffer( GLMmodel* model, const optix::Matrix4x4& transform )
{
  // create a buffer for the next-event estimation
  m_light_buffer = m_context->createBuffer( RT_BUFFER_INPUT );
  m_light_buffer->setFormat( RT_FORMAT_USER );
  m_light_buffer->setElementSize( sizeof( TriangleLight ) );

  // light sources
  unsigned int num_light = 0;
  extractAreaLights(model, transform, num_light, m_lights);  
  // write to the buffer
  m_light_buffer->setSize( 0 );
  if (num_light != 0)
  {
	m_light_buffer->setSize( num_light );
	memcpy( m_light_buffer->map(), &m_lights[0], num_light * sizeof( TriangleLight ) );
	m_light_buffer->unmap();
  }
  
}

void ObjLoader::getAreaLights(std::vector<TriangleLight> & lights)
{
	for(int i = 0; i < m_lights.size();i++)
	{
		lights.push_back(m_lights[i]); //copy
	}
}




